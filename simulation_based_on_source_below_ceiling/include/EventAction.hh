// ==============================================================
// EventAction.hh — Per-event accumulation for LUT-driven output
//
// Tracks:
//   1. Every particle entering a scintillator plate (LUT keys)
//   2. Whether that particle stopped in the plate
//   3. Total edep per plate (for coincidence cross-check)
//   4. Muon decays → tags decay-product secondaries
//
// Output per event:
//   scint_hits.csv — one row per particle×plate entry
//     (event_id, plate, pdg, track_id, ke_MeV, is_stopped,
//      time_ns, is_decay_product)
//
//   coincidence.csv — one row per event with ≥1 plate above threshold
//     (event_id, coinc_mask, n_fired, edep0..3_MeV)
// ==============================================================

#ifndef EVENT_ACTION_HH
#define EVENT_ACTION_HH

#include "G4UserEventAction.hh"
#include "G4Types.hh"

#include <set>
#include <map>
#include <string>
#include <vector>

// Discriminator threshold for coincidence cross-check (MeV)
static constexpr double kDiscrimThreshold_MeV = 0.5;

// ── Scintillator entry record (one per particle×plate crossing) ───────────
struct ScintHitRecord {
    G4int    plate;            // 0–3
    G4int    pdg;
    G4int    trackID;
    G4double ke_MeV;           // kinetic energy at entry
    G4double time_ns;          // global time at entry
    G4bool   isDecayProduct;   // parent muon decayed → this is e±
};

class EventAction : public G4UserEventAction
{
public:
    EventAction();
    ~EventAction() override = default;

    void BeginOfEventAction(const G4Event*) override;
    void EndOfEventAction(const G4Event*)   override;

    // ── Called from SteppingAction ──────────────────────────────────────────
    // Particle crossed into scintillator plate
    void RecordScintEntry(G4int plate, G4int pdg, G4int trackID,
                          G4double ke_MeV, G4double time_ns,
                          G4bool isDecayProduct);

    // Particle crossed out of scintillator plate
    void RecordScintExit(G4int plate, G4int trackID);

    // Accumulate energy deposit in scintillator (all species, for coinc)
    void AddScintEdep(G4int plate, G4double edep_MeV);

    // ── Called from SteppingAction / TrackingAction ─────────────────────────
    // A muon decayed — record its trackID so daughters can be tagged
    void RegisterDecayedMuon(G4int trackID);

    // Is this parent a muon that decayed?
    bool IsDecayMuonDaughter(G4int parentID) const {
        return fDecayedMuonIDs.count(parentID) > 0;
    }

private:
    std::vector<ScintHitRecord> fEntries;

    // Per-plate tracking: which trackIDs entered / exited
    std::set<G4int> fExited[4];

    // Total edep per plate (for coincidence threshold)
    G4double fEdep[4] = {};

    // TrackIDs of muons that underwent Decay (anywhere in the geometry)
    std::set<G4int> fDecayedMuonIDs;
};

#endif
