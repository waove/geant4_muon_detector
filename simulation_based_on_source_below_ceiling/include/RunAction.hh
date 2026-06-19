// ==============================================================
// RunAction.hh — Output management for LUT-driven simulation
//
// Two per-event CSV files:
//   scint_hits.csv    — LUT keys + timestamp for every scintillator entry
//   coincidence.csv   — edep-based coincidence cross-check
//
// One run-level summary:
//   coincidence_summary.csv — pattern counts, efficiencies, mean edep
// ==============================================================

#ifndef RUN_ACTION_HH
#define RUN_ACTION_HH

#include "G4UserRunAction.hh"
#include <fstream>

class RunAction : public G4UserRunAction
{
public:
    RunAction();
    ~RunAction() override = default;

    void BeginOfRunAction(const G4Run*) override;
    void EndOfRunAction(const G4Run*)   override;

    std::ofstream& GetHitsStream()  { return fHitsFile; }
    std::ofstream& GetCoincStream() { return fCoincFile; }

    // ── Coincidence accumulation (called from EventAction) ────────────────
    void AccumulateCoinc(int coinc, double edep0, double edep1,
                         double edep2, double edep3);

private:
    std::ofstream fHitsFile;
    std::ofstream fCoincFile;

    // Run-level coincidence counters (same logic as full sim)
    long   fCoincCount[16]   = {};
    double fCoincEdepSum[4]  = {};
    long   fCoincEdepN[4]    = {};
    long   fEffDenom[4]      = {};
    long   fEffNum           = 0;
};

#endif
