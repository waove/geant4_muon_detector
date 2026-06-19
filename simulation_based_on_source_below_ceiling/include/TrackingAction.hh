// ==============================================================
// TrackingAction.hh — Track-level hooks
//
// Catches decay products that are CREATED INSIDE a scintillator
// (muon stopped and decayed in the plate).  These secondaries
// never cross a geometric boundary, so SteppingAction's boundary
// logic misses them.  PreUserTrackingAction records their "entry"
// at birth position.
// ==============================================================

#ifndef TRACKING_ACTION_HH
#define TRACKING_ACTION_HH

#include "G4UserTrackingAction.hh"

class EventAction;

class TrackingAction : public G4UserTrackingAction
{
public:
    explicit TrackingAction(EventAction* ea);
    ~TrackingAction() override = default;

    void PreUserTrackingAction(const G4Track* track)  override;
    void PostUserTrackingAction(const G4Track* track) override;

private:
    EventAction* fEventAction;
};

#endif
