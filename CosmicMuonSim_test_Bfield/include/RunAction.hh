#ifndef RUN_ACTION_HH
#define RUN_ACTION_HH

#include "G4UserRunAction.hh"
#include "EventAction.hh"
#include <fstream>
#include <array>

class RunAction : public G4UserRunAction
{
public:
    RunAction();
    ~RunAction() override = default;

    void BeginOfRunAction(const G4Run*) override;
    void EndOfRunAction(const G4Run*)   override;

    std::ofstream& GetEdepStream()              { return fEdepPerEventFile; }
    std::ofstream& GetStepStream()              { return fStepFile; }
    std::ofstream& GetTrajStream()              { return fTrajFile; }
    std::ofstream& GetEntryKEStream()           { return fEntryKEFile; }
    std::ofstream& GetDecayStream()             { return fDecayFile; }
    std::ofstream& GetDecaySecEntryKEStream()   { return fDecaySecEntryKEFile; }
    std::ofstream& GetCoincStream()             { return fCoincFile; }
    std::ofstream& GetGunEnergyStream() const   { return fGunEnergyFile; }

    // ---- Event caps ----
    // ── Increased caps for B-field electron test ──
    // kNDetailEvents raised to 100 so we capture many spiral trajectories.
    static constexpr int kNDetailEvents    = 100;
    static constexpr int kMaxEdepEvents    = 10000;
    static constexpr int kMaxStepEvents    = 1000;
    static constexpr int kNGunEnergyEvents = 10000;

    G4int fNDetailedWritten = 0;
    G4int GetDetailCount()       const { return fNDetailedWritten; }
    void  IncrementDetailCount()       { ++fNDetailedWritten; }

    bool ShouldRecordGunEnergy(G4int eventID) const {
        return eventID >= kNDetailEvents
            && eventID <  kNDetailEvents + kNGunEnergyEvents;
    }

    void AccumulateEdep   (int vol, int part, double edep);
    void AccumulateStepLen(int vol, int part, double stepLen);
    void AccumulateTracking(int vol, int part, int nEntered, int nStopped);
    void AccumulateConditionalStopping(int vol, int part, int nStopped);
    void AccumulateDecay(int vol, int muonIdx);
    void AccumulateDecaySec(int vol, int part, int nEntered, int nStopped);

    // ── Coincidence accumulation ──────────────────────────────────────────
    // coinc  : 4-bit bitmask (bit i = Scint_i fired above threshold)
    // alFired: Al plate also above threshold
    // edep*  : individual plate edep in MeV (for mean edep per coincidence)
    void AccumulateCoinc(int coinc, bool alFired,
                         double edep0, double edep1,
                         double edep2, double edep3,
                         double edepAl);

private:
    std::ofstream fEdepPerEventFile;
    std::ofstream fStepFile;
    std::ofstream fTrajFile;
    std::ofstream fEntryKEFile;
    std::ofstream fDecayFile;
    std::ofstream fDecaySecEntryKEFile;
    std::ofstream fCoincFile;
    mutable std::ofstream fGunEnergyFile;

    double fTotalEdep   [kNVols][kNParts] = {};
    double fTotalStepLen[kNVols][kNParts] = {};
    long   fTotalEntered[kNVols][kNParts] = {};
    long   fTotalStopped[kNVols][kNParts] = {};
    long   fCondStopped [kNVols][kNParts] = {};
    long   fTotalDecays [kNVols][2]       = {};
    long   fDecSecEntered[kNVols][kNParts] = {};
    long   fDecSecStopped[kNVols][kNParts] = {};

    // ── Coincidence counters ──────────────────────────────────────────────
    // fCoincCount[mask] = number of events with exactly this 4-bit pattern
    // mask ranges 0–15; mask=0 means nothing fired.
    long   fCoincCount[16]   = {};

    // fCoincWithAl[mask] = events with scint pattern 'mask' AND Al fired
    long   fCoincWithAl[16]  = {};

    // Summed edep per plate for computing mean edep in coincident events
    // [plate 0–3], accumulated only when that plate is in the coincidence
    double fCoincEdepSum[4]  = {};
    long   fCoincEdepN  [4]  = {};

    // ── Efficiency counters ───────────────────────────────────────────────
    // Efficiency of plate X = events where ALL plates fire /
    //                         events where all plates EXCEPT X fire.
    // fEffDenom[x] = events where plates {0,1,2,3} \ {x} all fired
    // fEffNum  [x] = events where ALL four plates fired (same for all x)
    long   fEffDenom[4]      = {};  // 3-fold trigger (excluding plate x)
    long   fEffNum           = 0;   // 4-fold coincidence (all fire)

    static constexpr double kVolumeMM3[kNVols] = {
        20.*628.*1000., 20.*628.*1000., 20.*628.*1000., 20.*628.*1000.,
        10.*628.*1000.,
        190.*630.*720. - 188.*628.*720.,
        188.*628.*720. - 4*20.*628.*1000. - 10.*628.*1000.,
        200.*1600.*1600., 400.*1600.*1600., 0.
    };
};

#endif
