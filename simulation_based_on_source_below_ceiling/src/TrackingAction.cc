// ==============================================================
// TrackingAction.cc — Track creation/destruction hooks
//
// KEY ROLE: catch decay products born INSIDE a scintillator.
//
// When a muon stops and decays inside a scintillator plate, the
// decay electron/positron is created at that same position.  It
// never crosses a geometric boundary to "enter" the plate, so
// SteppingAction's boundary logic misses it entirely.
//
// PreUserTrackingAction fires when a new track is about to be
// propagated.  If that track:
//   (a) was created by "Decay",
//   (b) has a parent that is a known decayed muon, and
//   (c) starts inside a scintillator plate
// → we record it as a scintillator entry at its birth time.
//
// This ensures the "stop" signal for the lifetime measurement
// is captured regardless of where the muon decayed.
// ==============================================================

#include "TrackingAction.hh"
#include "EventAction.hh"

#include "G4Track.hh"
#include "G4ParticleDefinition.hh"
#include "G4VProcess.hh"
#include "G4VPhysicalVolume.hh"
#include "G4SystemOfUnits.hh"

static G4int ScintPlateIndex(const G4String& name)
{
    if (name == "Plate_scin_0") return 0;
    if (name == "Plate_scin_1") return 1;
    if (name == "Plate_scin_2") return 2;
    if (name == "Plate_scin_3") return 3;
    return -1;
}

// Must match both "Decay" and "DecayWithSpin" (G4SpinDecayPhysics)
static bool IsDecayProcess(const G4String& name)
{
    return name == "Decay" || name == "DecayWithSpin";
}

TrackingAction::TrackingAction(EventAction* ea)
    : fEventAction(ea)
{}

void TrackingAction::PreUserTrackingAction(const G4Track* track)
{
    // Only interested in secondaries (parentID > 0)
    G4int parentID = track->GetParentID();
    if (parentID == 0) return;

    // Only decay products (matches both "Decay" and "DecayWithSpin")
    const G4VProcess* creator = track->GetCreatorProcess();
    if (!creator || !IsDecayProcess(creator->GetProcessName())) return;

    // Only daughters of muons that decayed
    if (!fEventAction->IsDecayMuonDaughter(parentID)) return;

    // Check if born inside a scintillator
    const G4VPhysicalVolume* vol = track->GetVolume();
    if (!vol) return;

    G4int plate = ScintPlateIndex(vol->GetName());
    if (plate < 0) return;

    // Record as a scintillator entry at birth
    G4int    pdg     = track->GetDefinition()->GetPDGEncoding();
    G4int    trkID   = track->GetTrackID();
    G4double ke_MeV  = track->GetKineticEnergy() / MeV;
    G4double time_ns = track->GetGlobalTime() / ns;

    fEventAction->RecordScintEntry(
        plate, pdg, trkID, ke_MeV, time_ns, /*isDecayProduct=*/true);
}

void TrackingAction::PostUserTrackingAction(const G4Track* /*track*/)
{
    // Nothing needed — exit detection handled by SteppingAction
}
