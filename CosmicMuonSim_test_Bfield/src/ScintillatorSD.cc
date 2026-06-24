// ==============================================================
// ScintillatorSD.cc
//
// ProcessHits() is called for EVERY Geant4 step inside the
// scintillator logical volume this SD is attached to.
//
// ★ THIS IS YOUR MAIN CUSTOMISATION POINT ★
//
// Examples of things you can do here:
//   - Record only muons (filter by PDG)
//   - Record only primary particles (trackID == 1)
//   - Record only secondaries created by a specific process
//   - Threshold on energy deposit
//   - Accumulate total edep per event per plate (for trigger logic)
//   - Store full step-by-step trajectory
// ==============================================================

#include "ScintillatorSD.hh"

#include "G4Step.hh"
#include "G4Track.hh"
#include "G4HCofThisEvent.hh"
#include "G4SDManager.hh"
#include "G4ParticleDefinition.hh"
#include "G4VProcess.hh"
#include "G4SystemOfUnits.hh"

// ScintHit uses standard new/delete (no G4Allocator)

ScintillatorSD::ScintillatorSD(const G4String& name, G4int plateIndex)
    : G4VSensitiveDetector(name),
      fPlateIndex(plateIndex)
{
    collectionName.insert("ScintHC_" + std::to_string(plateIndex));
}

void ScintillatorSD::Initialize(G4HCofThisEvent* hce)
{
    fHitsCollection = new ScintHitsCollection(
        SensitiveDetectorName, collectionName[0]);

    if (fHCID < 0)
        fHCID = G4SDManager::GetSDMpointer()
                    ->GetCollectionID(fHitsCollection);
    hce->AddHitsCollection(fHCID, fHitsCollection);
}

// ==============================================================
// ProcessHits — called for every step in this scintillator
//
// CUSTOMISE THIS METHOD for your tracking needs.
// The example below records every step, but you can add
// any filtering logic you want.
// ==============================================================
G4bool ScintillatorSD::ProcessHits(G4Step* step, G4TouchableHistory*)
{
    G4double edep = step->GetTotalEnergyDeposit();

    // --- Example filter: skip steps with zero energy deposit ---
    // Remove this if you want to track all steps (e.g. for fluence)
    // if (edep <= 0.) return false;

    const G4Track*              track = step->GetTrack();
    const G4ParticleDefinition* part  = track->GetDefinition();
    const G4StepPoint*          pre   = step->GetPreStepPoint();

    // ============================================================
    //  ★ ADD YOUR PARTICLE FILTERS HERE ★
    //
    //  Examples (uncomment what you need):
    //
    //  // Only record muons:
    //  if (std::abs(part->GetPDGEncoding()) != 13) return false;
    //
    //  // Only record primary particle:
    //  if (track->GetTrackID() != 1) return false;
    //
    //  // Only record electrons/positrons from muon ionisation:
    //  if (std::abs(part->GetPDGEncoding()) != 11) return false;
    //  auto* proc = track->GetCreatorProcess();
    //  if (!proc || proc->GetProcessName() != "muIoni") return false;
    //
    //  // Only record particles above 1 MeV kinetic energy:
    //  if (pre->GetKineticEnergy() < 1.0*MeV) return false;
    //
    // ============================================================

    // --- Build the hit ---
    auto* hit = new ScintHit();
    hit->fPlateIndex   = fPlateIndex;
    hit->fTrackID      = track->GetTrackID();
    hit->fParentID     = track->GetParentID();
    hit->fPDG          = part->GetPDGEncoding();
    hit->fParticleName = part->GetParticleName();
    hit->fEdep         = edep;
    hit->fStepLength   = step->GetStepLength();
    hit->fKineticE     = pre->GetKineticEnergy();
    hit->fTime         = pre->GetGlobalTime();
    hit->fPosition     = pre->GetPosition();
    hit->fMomentumDir  = pre->GetMomentumDirection();

    // Creator process (empty for primaries)
    const G4VProcess* creatorProc = track->GetCreatorProcess();
    hit->fProcessName = creatorProc ? creatorProc->GetProcessName() : "primary";

    fHitsCollection->insert(hit);
    return true;
}

void ScintillatorSD::EndOfEvent(G4HCofThisEvent*)
{
    // Optional: print summary per event (useful for debugging)
    // G4int nHits = fHitsCollection->entries();
    // G4cout << "Plate " << fPlateIndex << ": " << nHits << " hits" << G4endl;
}
