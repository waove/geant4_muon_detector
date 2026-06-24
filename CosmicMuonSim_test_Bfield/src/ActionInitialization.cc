#include "ActionInitialization.hh"
#include "PrimaryGeneratorAction.hh"
#include "RunAction.hh"
#include "EventAction.hh"
#include "SteppingAction.hh"
#include "TrackingAction.hh"

void ActionInitialization::BuildForMaster() const
{
    SetUserAction(new RunAction());
}

void ActionInitialization::Build() const
{
    SetUserAction(new PrimaryGeneratorAction());

    auto* runAction   = new RunAction();
    auto* eventAction = new EventAction();

    SetUserAction(runAction);
    SetUserAction(eventAction);
    SetUserAction(new SteppingAction(eventAction));
    SetUserAction(new TrackingAction());
}
