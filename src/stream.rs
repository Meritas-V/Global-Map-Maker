//! Loading ground meshes in the background, the way old games streamed their worlds.
//!
//! The main thread asks for the patches it needs; worker threads make them (height tiles included), nearest
//! first; the main thread picks up finished ones a few at a time. Only meshes that already exist are drawn:
//! where a finer one is not ready, its coarser ancestor shows instead, so the frame never stalls and the
//! ground never has holes. A few coarse levels stay loaded for good as the floor under everything.

use crate::ground::{build_patch_from, PatchKey, PatchMesh};
use crate::pixelpaint::PixelId;
use crate::terrain::HeightSource;
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;

struct State {
    /// Waiting to be made, best (smallest priority number) last so workers can `pop`.
    pending: Vec<(f64, PatchKey)>,
    in_flight: HashSet<PatchKey>,
    done: Vec<(PatchKey, PatchMesh)>,
    /// Keys in `done`, so a mesh finished but not picked up yet is not asked for again.
    done_keys: HashSet<PatchKey>,
    quit: bool,
}

struct Shared {
    state: Mutex<State>,
    wake: Condvar,
    terrain: Arc<dyn HeightSource + Send + Sync>,
}

pub struct Streamer {
    shared: Arc<Shared>,
    threads: Vec<JoinHandle<()>>,
}

impl Streamer {
    pub fn new(terrain: Arc<dyn HeightSource + Send + Sync>, workers: usize) -> Streamer {
        let shared = Arc::new(Shared {
            state: Mutex::new(State { pending: Vec::new(), in_flight: HashSet::new(), done: Vec::new(), done_keys: HashSet::new(), quit: false }),
            wake: Condvar::new(),
            terrain,
        });
        let threads = (0..workers.max(1))
            .map(|i| {
                let sh = shared.clone();
                std::thread::Builder::new()
                    .name(format!("ground-loader-{i}"))
                    .spawn(move || worker(sh))
                    .expect("loader thread")
            })
            .collect();
        Streamer { shared, threads }
    }

    /// Replaces the wish list: what is wanted right now, each with a priority (smaller = sooner). Things
    /// asked for earlier and not wanted any more are dropped before they start.
    pub fn request(&self, mut wanted: Vec<(f64, PatchKey)>) {
        let mut st = self.shared.state.lock().unwrap();
        let mut seen = HashSet::new();
        wanted.retain(|(_, k)| !st.in_flight.contains(k) && !st.done_keys.contains(k) && seen.insert(*k));
        // Best last.
        wanted.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        st.pending = wanted;
        drop(st);
        self.shared.wake.notify_all();
    }

    /// Finished meshes (at most `max`).
    pub fn take_done(&self, max: usize) -> Vec<(PatchKey, PatchMesh)> {
        let mut st = self.shared.state.lock().unwrap();
        let n = st.done.len().min(max);
        let out: Vec<(PatchKey, PatchMesh)> = st.done.drain(..n).collect();
        for (k, _) in &out {
            st.done_keys.remove(k);
        }
        out
    }

    /// Meshes still waiting or being made, plus finished ones not picked up yet.
    pub fn busy(&self) -> usize {
        let st = self.shared.state.lock().unwrap();
        st.pending.len() + st.in_flight.len() + st.done.len()
    }
}

impl Drop for Streamer {
    fn drop(&mut self) {
        self.shared.state.lock().unwrap().quit = true;
        self.shared.wake.notify_all();
        for t in self.threads.drain(..) {
            let _ = t.join();
        }
    }
}

fn worker(sh: Arc<Shared>) {
    loop {
        let key = {
            let mut st = sh.state.lock().unwrap();
            loop {
                if st.quit {
                    return;
                }
                if let Some((_, k)) = st.pending.pop() {
                    st.in_flight.insert(k);
                    break k;
                }
                st = sh.wake.wait(st).unwrap();
            }
        };
        let mesh = build_patch_from(key, Some(&*sh.terrain));
        let mut st = sh.state.lock().unwrap();
        st.in_flight.remove(&key);
        st.done_keys.insert(key);
        st.done.push((key, mesh));
    }
}

/// Coarsest levels kept resident for good: whatever is missing below them, something is always there to show.
pub const BASE_LEVEL: u8 = 4;

/// What to draw this frame.
#[derive(Debug, Default)]
pub struct Plan {
    /// Meshes to draw: a cut through the quadtree with no overlaps and no holes. Each wanted patch is drawn
    /// itself if it is ready, otherwise by its nearest ready ancestor (a coarser version of the same ground).
    pub draw: Vec<PatchKey>,
    /// Wanted patches that are not ready yet.
    pub missing: Vec<PatchKey>,
    /// Every ancestor (below the always-resident base levels) of the wanted patches. They are what a coarser
    /// stand-in is made of, so they are loaded first (coarse to fine) and kept in the cache: without them the
    /// nearest ready ancestor of a missing patch can be a continent-sized mesh that hides everything under it.
    pub chain: Vec<PatchKey>,
    /// Base-level patches that nothing stands in for yet (only at a cold start or after a long jump): make them now.
    pub holes: Vec<PatchKey>,
}

/// Every ancestor of `keys` from just below `BASE_LEVEL` down to their parents, once each.
pub fn chain_of(keys: &[PatchKey], key_of: &dyn Fn(PixelId) -> PatchKey) -> Vec<PatchKey> {
    let mut seen: HashSet<PixelId> = HashSet::new();
    let mut out = Vec::new();
    for k in keys {
        let mut id = k.id;
        while id.level > BASE_LEVEL + 1 {
            id = id.ancestor(id.level - 1);
            if !seen.insert(id) {
                break; // it and everything above it is already in
            }
            out.push(key_of(id));
        }
    }
    out
}

/// Decides what to draw from what is wanted (`keys`) and what is ready (`ready`; `by_id` is the newest ready
/// mesh of each patch). Only meshes that exist are ever drawn: refinement never goes into data that is not
/// loaded, so the ground has no holes and no flicker while it loads. `key_of` gives the mesh key of any patch.
pub fn plan_frame(keys: &[PatchKey], ready: &dyn Fn(&PatchKey) -> bool, by_id: &HashMap<PixelId, PatchKey>, key_of: &dyn Fn(PixelId) -> PatchKey) -> Plan {
    let mut plan = Plan::default();
    let mut chosen: Vec<PatchKey> = Vec::with_capacity(keys.len());
    let mut seen_hole = HashSet::new();
    plan.chain = chain_of(keys, key_of);
    for k in keys {
        if ready(k) {
            chosen.push(*k);
            continue;
        }
        plan.missing.push(*k);
        let found = (0..k.id.level).rev().find_map(|lv| {
            let a = k.id.ancestor(lv);
            by_id.get(&a).filter(|ak| ready(ak)).map(|ak| (lv, *ak))
        });
        match found {
            Some((_, ak)) => {
                chosen.push(ak);
            }
            None => {
                let b = key_of(k.id.ancestor(BASE_LEVEL.min(k.id.level)));
                if seen_hole.insert(b) {
                    plan.holes.push(b);
                }
            }
        }
    }
    // Keep a mesh only if no coarser mesh in the list covers it (a ready child next to a missing sibling is
    // hidden under the parent until all of them are ready), and only one mesh per patch.
    let covered: HashSet<PixelId> = chosen.iter().map(|k| k.id).collect();
    let mut seen = HashSet::new();
    for k in chosen {
        if (0..k.id.level).any(|lv| covered.contains(&k.id.ancestor(lv))) {
            continue;
        }
        if seen.insert(k.id) {
            plan.draw.push(k);
        }
    }
    plan
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::terrain::Terrain;

    fn key(face: u8, level: u8, x: u32, y: u32) -> PatchKey {
        PatchKey { id: PixelId::new(face, level, x, y), n: 4, lvl: 6, plvl: 5 }
    }

    #[test]
    fn the_background_loader_makes_what_it_is_asked_for_nearest_first() {
        let st = Streamer::new(Arc::new(Terrain::new(7)), 2);
        let wanted: Vec<(f64, PatchKey)> = (0..12).map(|i| (i as f64, key(0, 6, i, 3))).collect();
        st.request(wanted.clone());
        let t0 = std::time::Instant::now();
        let mut got = Vec::new();
        while got.len() < 12 && t0.elapsed().as_secs() < 20 {
            got.extend(st.take_done(100));
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        assert_eq!(got.len(), 12, "loader too slow or stuck");
        assert!(got.iter().all(|(_, m)| !m.verts.is_empty()));
        assert_eq!(st.busy(), 0);
        // The very first ones asked (best priority) are made before the last ones, whichever worker gets them.
        let pos = |k: PatchKey| got.iter().position(|(g, _)| *g == k).unwrap();
        assert!(pos(wanted[0].1) < pos(wanted[11].1));
    }

    #[test]
    fn old_requests_are_dropped_when_the_wish_list_changes() {
        let st = Streamer::new(Arc::new(Terrain::new(7)), 1);
        st.request((0..400).map(|i| (i as f64, key(1, 8, i, 5))).collect());
        st.request(vec![(0.0, key(2, 8, 1, 1))]);
        let t0 = std::time::Instant::now();
        let mut made = 0;
        while st.busy() > 0 && t0.elapsed().as_secs() < 20 {
            made += st.take_done(1000).len();
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        made += st.take_done(1000).len();
        assert!(made < 50, "{made} made: the dropped requests were still built");
        // Teardown joins the workers without hanging.
        drop(st);
    }

    #[test]
    fn a_missing_patch_is_covered_by_the_nearest_ready_ancestor_without_overlap() {
        let mk = |id: PixelId| PatchKey { id, n: 4, lvl: 6, plvl: 5 };
        let parent = key(0, 4, 3, 3);
        let kids: Vec<PatchKey> = parent.id.children().iter().map(|id| mk(*id)).collect();
        let key_of = |id: PixelId| mk(id);
        // Parent ready, kids wanted but not ready: the parent is drawn once.
        let by_id: HashMap<PixelId, PatchKey> = [(parent.id, parent)].into_iter().collect();
        let plan = plan_frame(&kids, &|k| *k == parent, &by_id, &key_of);
        assert_eq!(plan.draw, vec![parent]);
        assert_eq!(plan.missing.len(), 4);
        assert!(plan.holes.is_empty());
        // Three kids ready, one missing: the ready kids are hidden under the parent (no overlap, no hole).
        let ready_set: Vec<PatchKey> = vec![parent, kids[0], kids[1], kids[2]];
        let by_id: HashMap<PixelId, PatchKey> = ready_set.iter().map(|k| (k.id, *k)).collect();
        let plan = plan_frame(&kids, &|k| ready_set.contains(k), &by_id, &key_of);
        assert_eq!(plan.draw, vec![parent]);
        // All kids ready: the kids are drawn, not the parent.
        let ready_set: Vec<PatchKey> = vec![parent, kids[0], kids[1], kids[2], kids[3]];
        let by_id: HashMap<PixelId, PatchKey> = ready_set.iter().map(|k| (k.id, *k)).collect();
        let plan = plan_frame(&kids, &|k| ready_set.contains(k), &by_id, &key_of);
        assert_eq!(plan.draw.len(), 4);
        assert!(plan.missing.is_empty());
        // A deep patch with only a distant ancestor: that ancestor stands in and the next step down is asked for.
        let deep = key(0, 8, 40, 40);
        let anc = key_of(deep.id.ancestor(5));
        let by_id: HashMap<PixelId, PatchKey> = [(anc.id, anc)].into_iter().collect();
        let plan = plan_frame(&[deep], &|k| *k == anc, &by_id, &key_of);
        assert_eq!(plan.draw, vec![anc]);
        assert!(plan.chain.contains(&key_of(deep.id.ancestor(6))) && plan.chain.contains(&key_of(deep.id.ancestor(7))));
        assert!(!plan.chain.contains(&key_of(deep.id.ancestor(BASE_LEVEL))));
        // Nothing around: the base patch is asked for as a hole.
        let nothing: HashMap<PixelId, PatchKey> = HashMap::new();
        let plan = plan_frame(&[deep], &|_| false, &nothing, &key_of);
        assert_eq!(plan.holes, vec![key_of(deep.id.ancestor(BASE_LEVEL))]);
        assert!(plan.draw.is_empty());
    }
}
