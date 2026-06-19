// ==============================================================
// EventAction.cc — Per-event accumulation + output
//
// At end-of-event:
//   1. Resolve is_stopped for each scintillator entry
//      (entered but never exited the plate during this event)
//   2. Write LUT keys + timestamp to scint_hits.csv
//   3. Compute coincidence bitmask from plate edep
//   4. Write coincidence record to coincidence.csv
// ==============================================================

#include "EventAction.hh"
#include "RunAction.hh"

#include "G4Event.hh"
#include "G4RunManager.hh"
#include "G4SystemOfUnits.hh"

#include <cstring>

EventAction::EventAction() {}

void EventAction::BeginOfEventAction(const G4Event*)
{
    fEntries.clear();
    for (int p = 0; p < 4; ++p) {
        fExited[p].clear();
        fEdep[p] = 0.;
    }
    fDecayedMuonIDs.clear();
}

// ── Accumulation methods (called from SteppingAction / TrackingAction) ────

void EventAction::RecordScintEntry(G4int plate, G4int pdg, G4int trackID,
                                    G4double ke_MeV, G4double time_ns,
                                    G4bool isDecayProduct)
{
    fEntries.push_back({plate, pdg, trackID, ke_MeV, time_ns, isDecayProduct});
}

void EventAction::RecordScintExit(G4int plate, G4int trackID)
{
    fExited[plate].insert(trackID);
}

void EventAction::AddScintEdep(G4int plate, G4double edep_MeV)
{
    fEdep[plate] += edep_MeV;
}

void EventAction::RegisterDecayedMuon(G4int trackID)
{
    fDecayedMuonIDs.insert(trackID);
}

// ==============================================================
// EndOfEventAction
// ==============================================================
void EventAction::EndOfEventAction(const G4Event* event)
{
    G4int eventID = event->GetEventID();

    auto* runAction = const_cast<RunAction*>(
        static_cast<const RunAction*>(
            G4RunManager::GetRunManager()->GetUserRunAction()));

    // ── 1. Write scint_hits.csv ───────────────────────────────────────────
    if (!fEntries.empty()) {
        auto& hf = runAction->GetHitsStream();
        for (const auto& e : fEntries) {
            // is_stopped: entered this plate but never exited
            G4bool stopped = (fExited[e.plate].count(e.trackID) == 0);

            hf << eventID          << ","
               << e.plate          << ","
               << e.pdg            << ","
               << e.trackID        << ","
               << e.ke_MeV         << ","
               << (int)stopped     << ","
               << e.time_ns        << ","
               << (int)e.isDecayProduct << "\n";
        }
    }

    // ── 2. Coincidence cross-check ────────────────────────────────────────
    G4int coinc = 0;
    for (int s = 0; s < 4; ++s)
        if (fEdep[s] >= kDiscrimThreshold_MeV)
            coinc |= (1 << s);

    if (coinc > 0) {
        G4int nFired = __builtin_popcount(static_cast<unsigned>(coinc));

        runAction->GetCoincStream()
            << eventID << ","
            << coinc   << ","
            << nFired  << ","
            << fEdep[0] << ","
            << fEdep[1] << ","
            << fEdep[2] << ","
            << fEdep[3] << "\n";
    }

    // ── 3. Run-level coincidence accumulation ─────────────────────────────
    runAction->AccumulateCoinc(coinc, fEdep[0], fEdep[1], fEdep[2], fEdep[3]);

    if (eventID % 10000 == 0)
        G4cout << "Event " << eventID << " done" << G4endl;
}
