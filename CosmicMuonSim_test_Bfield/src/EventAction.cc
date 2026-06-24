// ==============================================================
// EventAction.cc
// ==============================================================

#include "EventAction.hh"
#include "RunAction.hh"
#include "ScintHit.hh"

#include "G4Event.hh"
#include "G4HCofThisEvent.hh"
#include "G4SDManager.hh"
#include "G4RunManager.hh"
#include "G4SystemOfUnits.hh"

#include <cstring>
#include <sstream>

const char* kVolNames[kNVols] = {
    "Scint_0", "Scint_1", "Scint_2", "Scint_3",
    "Al_plate", "Cu_walls", "Air_hole",
    "Ceiling", "Roof", "World"
};
const char* kPartNames[kNParts] = {
    "mu-", "mu+", "e-", "e+", "gamma", "other", "all"
};

VolIdx VolumeNameToIdx(const std::string& name)
{
    if (name == "Plate_scin_0") return kScint0;
    if (name == "Plate_scin_1") return kScint1;
    if (name == "Plate_scin_2") return kScint2;
    if (name == "Plate_scin_3") return kScint3;
    if (name == "Plate_Al")     return kAlPlate;
    if (name == "Shape1")       return kCopper;
    if (name == "Shape2")       return kAirHole;
    if (name == "Ceiling")      return kCeiling;
    if (name == "Roof")         return kRoof;
    return kWorld;
}

PartIdx PDGToPartIdx(G4int pdg)
{
    switch (pdg) {
        case  13: return kMuMinus;
        case -13: return kMuPlus;
        case  11: return kElectron;
        case -11: return kPositron;
        case  22: return kGamma;
        default:  return kOther;
    }
}

EventAction::EventAction() {}

void EventAction::BeginOfEventAction(const G4Event*)
{
    std::memset(fEdep,    0, sizeof(fEdep));
    std::memset(fStepLen, 0, sizeof(fStepLen));
    for (int v = 0; v < kNVols; ++v)
        for (int p = 0; p < kNParts; ++p) {
            fEntered[v][p].clear();
            fExited [v][p].clear();
            fDecSecEntered[v][p].clear();
            fDecSecExited [v][p].clear();
        }
    fScintEntries.clear();
    fDecayRecords.clear();
    fDecaySecEntryKEs.clear();
    fAlDecayMuonTrackIDs.clear();
    fHitDetector = false;
    fTrajBuffer.clear();
}

void EventAction::AccumulateStep(VolIdx vol, PartIdx part,
                                  G4double edep, G4double stepLen)
{
    fEdep   [vol][part] += edep;
    fEdep   [vol][kAll] += edep;
    fStepLen[vol][part] += stepLen;
    fStepLen[vol][kAll] += stepLen;
}

void EventAction::RecordEntry(VolIdx vol, PartIdx part, G4int trackID)
{
    fEntered[vol][part].insert(trackID);
    fEntered[vol][kAll].insert(trackID);
}

void EventAction::RecordExit(VolIdx vol, PartIdx part, G4int trackID)
{
    fExited[vol][part].insert(trackID);
    fExited[vol][kAll].insert(trackID);
}

void EventAction::RecordScintEntry(VolIdx vol, G4int pdg,
                                    G4int trackID, G4double ke)
{
    fScintEntries.push_back({(G4int)vol, pdg, trackID, ke / MeV});
}

void EventAction::RecordDecay(VolIdx vol, G4int pdg, G4int trackID,
                               G4double ke, G4double x, G4double y,
                               G4double z, G4double time)
{
    fDecayRecords.push_back({(G4int)vol, pdg, trackID,
                              ke / MeV,
                              x / mm, y / mm, z / mm,
                              time / ns});
    if (vol == kAlPlate)
        fAlDecayMuonTrackIDs.insert(trackID);
}

void EventAction::RecordDecaySecEntry(VolIdx vol, PartIdx part, G4int trackID)
{
    fDecSecEntered[vol][part].insert(trackID);
    fDecSecEntered[vol][kAll].insert(trackID);
}

void EventAction::RecordDecaySecExit(VolIdx vol, PartIdx part, G4int trackID)
{
    fDecSecExited[vol][part].insert(trackID);
    fDecSecExited[vol][kAll].insert(trackID);
}

void EventAction::RecordDecaySecEntryKE(VolIdx vol, G4int pdg,
                                         G4int trackID, G4double ke)
{
    fDecaySecEntryKEs.push_back({(G4int)vol, pdg, trackID, ke / MeV});
}

void EventAction::EndOfEventAction(const G4Event* event)
{
    G4int eventID = event->GetEventID();

    auto* runAction = const_cast<RunAction*>(
        static_cast<const RunAction*>(
            G4RunManager::GetRunManager()->GetUserRunAction()));

    // ==============================================================
    // Coincidence logic
    //
    // Build a 4-bit bitmask of which scintillator plates fired above
    // the discriminator threshold this event.
    //
    //   bit 0 = Scint_0,  bit 1 = Scint_1,
    //   bit 2 = Scint_2,  bit 3 = Scint_3
    //
    // fEdep[][kAll] contains total deposited energy from all species.
    // ==============================================================
    G4int coinc = 0;
    for (int s = 0; s < 4; ++s) {
        G4double edep_MeV = fEdep[s][kAll] / MeV;
        if (edep_MeV >= kDiscrimThreshold_MeV)
            coinc |= (1 << s);
    }

    // Count bits set — gives the fold of the coincidence
    G4int nFired = __builtin_popcount(static_cast<unsigned>(coinc));

    // Also check whether the Al plate fired (useful for stopping studies)
    bool alFired = (fEdep[kAlPlate][kAll] / MeV) >= kDiscrimThreshold_MeV;

    // Individual plate edep for the CSV (all species, in MeV)
    G4double edep0 = fEdep[kScint0][kAll] / MeV;
    G4double edep1 = fEdep[kScint1][kAll] / MeV;
    G4double edep2 = fEdep[kScint2][kAll] / MeV;
    G4double edep3 = fEdep[kScint3][kAll] / MeV;
    G4double edepAl = fEdep[kAlPlate][kAll] / MeV;

    // Accumulate run-level coincidence statistics
    runAction->AccumulateCoinc(coinc, alFired,
                                edep0, edep1, edep2, edep3, edepAl);

    // Write per-event coincidence record (no cap)
    if (coinc > 0 || alFired) {
        runAction->GetCoincStream()
            << eventID  << ","
            << coinc    << ","
            << nFired   << ","
            << (int)alFired << ","
            << edep0    << ","
            << edep1    << ","
            << edep2    << ","
            << edep3    << ","
            << edepAl   << "\n";
    }

    // ==============================================================
    // Global run-level accumulation
    // ==============================================================
    for (int v = 0; v < kNVols; ++v) {
        for (int p = 0; p < kNParts; ++p) {
            runAction->AccumulateEdep(v, p, fEdep[v][p]);
            runAction->AccumulateStepLen(v, p, fStepLen[v][p]);

            G4int nEntered = (G4int)fEntered[v][p].size();
            G4int nStopped = 0;
            for (G4int trkID : fEntered[v][p]) {
                if (fExited[v][p].find(trkID) == fExited[v][p].end())
                    ++nStopped;
            }
            runAction->AccumulateTracking(v, p, nEntered, nStopped);
        }
    }

    // ==============================================================
    // Conditional stopping
    // ==============================================================
    for (int p = 0; p < kNParts; ++p) {
        const auto& s0tracks = fEntered[kScint0][p];
        if (s0tracks.empty()) continue;
        for (int v = 0; v < kNVols; ++v) {
            if (v == kScint0) continue;
            G4int nCondStopped = 0;
            for (G4int trkID : s0tracks) {
                if (fEntered[v][p].count(trkID) > 0 &&
                    fExited [v][p].count(trkID) == 0)
                    ++nCondStopped;
            }
            if (nCondStopped > 0)
                runAction->AccumulateConditionalStopping(v, p, nCondStopped);
        }
    }

    // ==============================================================
    // Decay secondary stopping
    // ==============================================================
    if (!fAlDecayMuonTrackIDs.empty()) {
        for (int v = 0; v < kNVols; ++v) {
            for (int p = 0; p < kNParts; ++p) {
                G4int nEntered = (G4int)fDecSecEntered[v][p].size();
                G4int nStopped = 0;
                for (G4int trkID : fDecSecEntered[v][p]) {
                    if (fDecSecExited[v][p].find(trkID) ==
                        fDecSecExited[v][p].end())
                        ++nStopped;
                }
                if (nEntered > 0)
                    runAction->AccumulateDecaySec(v, p, nEntered, nStopped);
            }
        }
    }

    // ==============================================================
    // entry_ke.csv
    // ==============================================================
    {
        auto& entryFile = runAction->GetEntryKEStream();
        for (const auto& r : fScintEntries)
            entryFile << eventID   << ","
                      << r.vol     << ","
                      << r.pdg     << ","
                      << r.trackID << ","
                      << r.ke_MeV  << "\n";
    }

    // ==============================================================
    // decay_secondary_entry_ke.csv
    // ==============================================================
    if (!fDecaySecEntryKEs.empty()) {
        auto& secFile = runAction->GetDecaySecEntryKEStream();
        for (const auto& r : fDecaySecEntryKEs)
            secFile << eventID    << ","
                    << kVolNames[r.vol] << ","
                    << r.pdg      << ","
                    << r.trackID  << ","
                    << r.ke_MeV   << "\n";
    }

    // ==============================================================
    // decay.csv
    // ==============================================================
    if (!fDecayRecords.empty()) {
        auto& decayFile = runAction->GetDecayStream();
        for (const auto& d : fDecayRecords) {
            decayFile << eventID    << ","
                      << kVolNames[d.vol] << ","
                      << d.pdg     << ","
                      << d.trackID << ","
                      << d.ke_MeV  << ","
                      << d.x_mm    << ","
                      << d.y_mm    << ","
                      << d.z_mm    << ","
                      << d.time_ns << "\n";
            runAction->AccumulateDecay(d.vol, d.pdg == 13 ? 0 : 1);
        }
    }

    // ==============================================================
    // edep_per_event.csv — capped
    // ==============================================================
    if (eventID < RunAction::kMaxEdepEvents) {
        auto& edepFile = runAction->GetEdepStream();
        for (int v = 0; v < kNVols; ++v) {
            edepFile << eventID << "," << kVolNames[v];
            for (int p = 0; p < kNParts; ++p)
                edepFile << "," << fEdep[v][p] / MeV;
            edepFile << "\n";
        }
    }

    // ==============================================================
    // steps.csv — capped
    // ==============================================================
    if (eventID < RunAction::kMaxStepEvents) {
        auto* hce = event->GetHCofThisEvent();
        if (hce) {
            auto& stepFile = runAction->GetStepStream();
            for (int p = 0; p < 4; ++p) {
                if (fHCIDs[p] < 0) {
                    G4String hcName = "ScintSD_" + std::to_string(p)
                                    + "/ScintHC_" + std::to_string(p);
                    fHCIDs[p] = G4SDManager::GetSDMpointer()
                                    ->GetCollectionID(hcName);
                }
                if (fHCIDs[p] < 0) continue;
                auto* hc = dynamic_cast<G4THitsCollection<ScintHit>*>(
                               hce->GetHC(fHCIDs[p]));
                if (!hc) continue;
                G4int nHits = hc->entries();
                for (G4int i = 0; i < nHits; ++i) {
                    const auto* hit = (*hc)[i];
                    stepFile << eventID                  << ","
                             << hit->fPlateIndex         << ","
                             << hit->fTrackID            << ","
                             << hit->fParentID           << ","
                             << hit->fPDG                << ","
                             << hit->fParticleName        << ","
                             << hit->fProcessName         << ","
                             << hit->fEdep / MeV          << ","
                             << hit->fStepLength / mm     << ","
                             << hit->fKineticE / MeV      << ","
                             << hit->fPosition.x() / mm   << ","
                             << hit->fPosition.y() / mm   << ","
                             << hit->fPosition.z() / mm   << ","
                             << hit->fMomentumDir.x()     << ","
                             << hit->fMomentumDir.y()     << ","
                             << hit->fMomentumDir.z()     << ","
                             << hit->fTime / ns           << "\n";
                }
            }
        }
    }

    // ==============================================================
    // trajectories.csv — capped
    // ==============================================================
    if (fHitDetector && runAction->GetDetailCount() < RunAction::kNDetailEvents) {
        auto& tf = runAction->GetTrajStream();
        for (const auto& line : fTrajBuffer)
            tf << line;
        runAction->IncrementDetailCount();
    }

    if (eventID % 10000 == 0)
        G4cout << "Event " << eventID << " done" << G4endl;
}
