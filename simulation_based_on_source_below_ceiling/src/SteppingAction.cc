// ==============================================================
// SteppingAction.cc — Global step-level hook (simplified)
//
// For every step in the simulation:
//   1. If step is inside a scintillator → accumulate edep
//   2. If entering a scintillator → record entry (LUT key)
//   3. If exiting  a scintillator → record exit  (for is_stopped)
//   4. If muon decay process fires → register decayed muon
//   5. Decay-product entries are tagged via parent lookup
// ==============================================================

#include "SteppingAction.hh"
#include "EventAction.hh"

#include "G4Step.hh"
#include "G4Track.hh"
#include "G4StepPoint.hh"
#include "G4VPhysicalVolume.hh"
#include "G4ParticleDefinition.hh"
#include "G4VProcess.hh"
#include "G4SystemOfUnits.hh"

// ── Volume identification helpers ─────────────────────────────────────────
// Returns plate index 0–3 for scintillator volumes, -1 otherwise.
static G4int ScintPlateIndex(const G4String& name)
{
    if (name == "Plate_scin_0") return 0;
    if (name == "Plate_scin_1") return 1;
    if (name == "Plate_scin_2") return 2;
    if (name == "Plate_scin_3") return 3;
    return -1;
}

// ── Process name helper ───────────────────────────────────────────────────
// G4Decay        uses process name "Decay"
// G4DecayWithSpin uses process name "DecayWithSpin"
// Must match BOTH so the code works with or without G4SpinDecayPhysics.
static bool IsDecayProcess(const G4String& name)
{
    return name == "Decay" || name == "DecayWithSpin";
}

SteppingAction::SteppingAction(EventAction* ea)
    : fEventAction(ea)
{}

void SteppingAction::UserSteppingAction(const G4Step* step)
{
    const G4StepPoint* pre  = step->GetPreStepPoint();
    const G4StepPoint* post = step->GetPostStepPoint();

    G4VPhysicalVolume* preVol = pre->GetPhysicalVolume();
    if (!preVol) return;

    const G4String& volName = preVol->GetName();
    G4int plate = ScintPlateIndex(volName);

    const G4Track* track = step->GetTrack();
    G4int   pdg   = track->GetDefinition()->GetPDGEncoding();
    G4int   trkID = track->GetTrackID();

    // ── 1. Edep accumulation in scintillators ─────────────────────────────
    if (plate >= 0) {
        G4double edep = step->GetTotalEnergyDeposit() / MeV;
        if (edep > 0.)
            fEventAction->AddScintEdep(plate, edep);
    }

    // ── 2. Scintillator boundary entry ────────────────────────────────────
    if (plate >= 0 && pre->GetStepStatus() == fGeomBoundary) {
        G4double ke_MeV  = pre->GetKineticEnergy() / MeV;
        G4double time_ns = pre->GetGlobalTime() / ns;

        // IDENTIFY DECAY PRODUCTS
        // Must match both "Decay" (standard) and "DecayWithSpin"
        // (when G4SpinDecayPhysics is registered).
        //
        // DO NOT include muMinusCaptureAtRest — capture products are
        // nuclear de-excitation electrons, not muon decay electrons.
        // They come out at τ_capture ≈ 0.88 µs on Al, contaminating
        // the lifetime measurement and pulling τ_fit below 2.2 µs.
        G4bool isDecay = false;
        const G4VProcess* creator = track->GetCreatorProcess();
        
        if (creator && IsDecayProcess(creator->GetProcessName())) {
            if (pdg == 11 || pdg == -11) {
                isDecay = true;
            }
        }

        // Record the entry WITH the correct isDecay status
        fEventAction->RecordScintEntry(
            plate, pdg, trkID, ke_MeV, time_ns, isDecay);
    }

    // ── 3. Scintillator boundary exit ─────────────────────────────────────
    if (plate >= 0 && post->GetStepStatus() == fGeomBoundary) {
        fEventAction->RecordScintExit(plate, trkID);
    }

    // ── 4. Muon decay registration ──────────────────────────────────────
    // Register when the muon itself disappears via decay.
    // With G4SpinDecayPhysics the process name is "DecayWithSpin".
    if (pdg == 13 || pdg == -13) {
        const G4VProcess* endProc = post->GetProcessDefinedStep();
        if (endProc && IsDecayProcess(endProc->GetProcessName())) {
            fEventAction->RegisterDecayedMuon(trkID);
        }
    }
}