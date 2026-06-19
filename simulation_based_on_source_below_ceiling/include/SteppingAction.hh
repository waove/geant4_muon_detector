// ==============================================================
// SteppingAction.hh — Global step-level hook
//
// Handles:
//   - Scintillator boundary entry/exit detection
//   - Edep accumulation per scintillator plate
//   - Muon decay detection (any volume)
//   - Decay-product tagging at scintillator boundaries
// ==============================================================

#ifndef STEPPING_ACTION_HH
#define STEPPING_ACTION_HH

#include "G4UserSteppingAction.hh"

class EventAction;

class SteppingAction : public G4UserSteppingAction
{
public:
    explicit SteppingAction(EventAction* ea);
    ~SteppingAction() override = default;

    void UserSteppingAction(const G4Step* step) override;

private:
    EventAction* fEventAction;
};

#endif
