// ==============================================================
// SteppingAction.cc
// ==============================================================

#include "SteppingAction.hh"
#include "EventAction.hh"
#include "RunAction.hh"

#include "G4Step.hh"
#include "G4Track.hh"
#include "G4StepPoint.hh"
#include "G4VPhysicalVolume.hh"
#include "G4ParticleDefinition.hh"
#include "G4VProcess.hh"
#include "G4RunManager.hh"
#include "G4Event.hh"
#include "G4SystemOfUnits.hh"

#include <sstream>

SteppingAction::SteppingAction(EventAction* eventAction)
    : fEventAction(eventAction)
{}

void SteppingAction::UserSteppingAction(const G4Step* step)
{
    const G4StepPoint* pre  = step->GetPreStepPoint();
    const G4StepPoint* post = step->GetPostStepPoint();

    G4VPhysicalVolume* preVol = pre->GetPhysicalVolume();
    if (!preVol) return;

    VolIdx  vol   = VolumeNameToIdx(preVol->GetName());
    G4int   pdg   = step->GetTrack()->GetDefinition()->GetPDGEncoding();
    PartIdx part  = PDGToPartIdx(pdg);
    G4int   trkID = step->GetTrack()->GetTrackID();

    G4double edep    = step->GetTotalEnergyDeposit();
    G4double stepLen = step->GetStepLength();

    if (edep > 0. || stepLen > 0.)
        fEventAction->AccumulateStep(vol, part, edep, stepLen);

    // ── Global boundary detection ─────────────────────────────────────────
    if (pre->GetStepStatus() == fGeomBoundary) {
        fEventAction->RecordEntry(vol, part, trkID);

        // Muon entry KE — scintillators AND Al plate
        // Both use the same entry_ke.csv so plots can compare the muon
        // spectrum arriving at the Al absorber vs each scint plate.
        if ((vol >= kScint0 && vol <= kScint3 || vol == kAlPlate)
            && (pdg == 13 || pdg == -13))
        {
            fEventAction->RecordScintEntry(vol, pdg, trkID,
                                           pre->GetKineticEnergy());
        }
    }

    if (post->GetStepStatus() == fGeomBoundary)
        fEventAction->RecordExit(vol, part, trkID);

    // ── Muon decay detection ──────────────────────────────────────────────
    if (pdg == 13 || pdg == -13) {
        const G4VProcess* endProc = post->GetProcessDefinedStep();
        if (endProc && endProc->GetProcessName() == "Decay") {
            fEventAction->RecordDecay(
                vol, pdg, trkID,
                pre->GetKineticEnergy(),
                pre->GetPosition().x(),
                pre->GetPosition().y(),
                pre->GetPosition().z(),
                pre->GetGlobalTime());
        }
    }

    // ── Decay-secondary boundary tracking ────────────────────────────────
    {
        const G4Track*    track    = step->GetTrack();
        G4int             parentID = track->GetParentID();
        const G4VProcess* creator  = track->GetCreatorProcess();

        if (creator
            && creator->GetProcessName() == "Decay"
            && fEventAction->IsAlDecaySecondary(parentID))
        {
            if (pre->GetStepStatus() == fGeomBoundary) {
                // Stopping logic
                fEventAction->RecordDecaySecEntry(vol, part, trkID);

                // Entry KE for every volume the secondary crosses into
                fEventAction->RecordDecaySecEntryKE(vol, pdg, trkID,
                                                     pre->GetKineticEnergy());
            }

            if (post->GetStepStatus() == fGeomBoundary)
                fEventAction->RecordDecaySecExit(vol, part, trkID);
        }
    }

    // ── Flag detector hit ─────────────────────────────────────────────────
    if (vol == kScint0 || vol == kScint1 || vol == kScint2 || vol == kScint3
        || vol == kAlPlate || vol == kCopper || vol == kAirHole)
        fEventAction->SetHitDetector();

    // ── Trajectory buffering ──────────────────────────────────────────────
    auto* runAction = static_cast<const RunAction*>(
        G4RunManager::GetRunManager()->GetUserRunAction());
    if (runAction->GetDetailCount() >= RunAction::kNDetailEvents) return;

    G4int eventID = G4RunManager::GetRunManager()
                        ->GetCurrentEvent()->GetEventID();
    const G4Track* track = step->GetTrack();

    std::ostringstream oss;
    oss << eventID                                      << ","
        << trkID                                        << ","
        << track->GetParentID()                         << ","
        << pdg                                          << ","
        << track->GetDefinition()->GetParticleName()    << ","
        << kVolNames[vol]                               << ","
        << edep / MeV                                   << ","
        << stepLen / mm                                  << ","
        << pre->GetKineticEnergy() / MeV                << ","
        << pre->GetPosition().x() / mm                  << ","
        << pre->GetPosition().y() / mm                  << ","
        << pre->GetPosition().z() / mm                  << ","
        << pre->GetMomentumDirection().x()              << ","
        << pre->GetMomentumDirection().y()              << ","
        << pre->GetMomentumDirection().z()              << ","
        << pre->GetGlobalTime() / ns                    << "\n";

    fEventAction->BufferTrajLine(oss.str());
}
