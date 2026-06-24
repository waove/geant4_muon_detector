// ==============================================================
// TrackingAction.hh — Track-level hooks
//
// Called once when a track is created (PreUserTrackingAction)
// and once when it ends (PostUserTrackingAction).
//
// Use for:
//   - Recording what secondaries are created
//   - Logging muon decay products
//   - Selectively storing/killing tracks before they propagate
// ==============================================================

#ifndef TRACKING_ACTION_HH
#define TRACKING_ACTION_HH

#include "G4UserTrackingAction.hh"

class TrackingAction : public G4UserTrackingAction
{
public:
    TrackingAction()  = default;
    ~TrackingAction() override = default;

    void PreUserTrackingAction(const G4Track* track) override;
    void PostUserTrackingAction(const G4Track* track) override;
};

#endif
