// ==============================================================
// SteppingAction.hh — Global step-level hook
//
// This is called for EVERY step in the ENTIRE simulation,
// not just in sensitive detectors. Use it for:
//   - Tracking particles across all volumes
//   - Killing particles you don't want to track
//   - Collecting global statistics
//   - Debugging trajectory issues
// ==============================================================

#ifndef STEPPING_ACTION_HH
#define STEPPING_ACTION_HH

#include "G4UserSteppingAction.hh"

class EventAction;

class SteppingAction : public G4UserSteppingAction
{
public:
    explicit SteppingAction(EventAction* eventAction);
    ~SteppingAction() override = default;

    void UserSteppingAction(const G4Step* step) override;

private:
    EventAction* fEventAction;
};

#endif
