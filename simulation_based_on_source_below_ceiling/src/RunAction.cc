// ==============================================================
// RunAction.cc — Output file management + coincidence summary
// ==============================================================

#include "RunAction.hh"
#include "EventAction.hh"

#include "G4Run.hh"
#include "G4SystemOfUnits.hh"

#include <filesystem>
#include <iomanip>
#include <cstring>
#include <cmath>

RunAction::RunAction() {}

static std::string MaskLabel(int mask)
{
    std::string s;
    for (int i = 0; i < 4; ++i)
        if (mask & (1 << i)) {
            if (!s.empty()) s += "+";
            s += "S" + std::to_string(i);
        }
    return s.empty() ? "none" : s;
}

void RunAction::BeginOfRunAction(const G4Run*)
{
    std::filesystem::create_directories("results");

    std::memset(fCoincCount,   0, sizeof(fCoincCount));
    std::memset(fCoincEdepSum, 0, sizeof(fCoincEdepSum));
    std::memset(fCoincEdepN,   0, sizeof(fCoincEdepN));
    std::memset(fEffDenom,     0, sizeof(fEffDenom));
    fEffNum = 0;

    // ── scint_hits.csv ────────────────────────────────────────────────────
    // One row per particle entering a scintillator plate.
    // These are the LUT lookup keys + the timestamp for lifetime.
    fHitsFile.open("results/scint_hits.csv");
    fHitsFile << "event_id,plate,pdg,track_id,ke_MeV,"
              << "is_stopped,time_ns,is_decay_product\n";

    // ── coincidence.csv ───────────────────────────────────────────────────
    // One row per event where at least one plate fired above threshold.
    // Cross-reference only — downstream analysis uses LUT pulse heights.
    fCoincFile.open("results/coincidence.csv");
    fCoincFile << "event_id,coinc_mask,n_fired,"
               << "edep0_MeV,edep1_MeV,edep2_MeV,edep3_MeV\n";

    G4cout << "\n  Discriminator threshold: " << kDiscrimThreshold_MeV
           << " MeV (coincidence cross-check)" << G4endl;
}

void RunAction::AccumulateCoinc(int coinc,
                                 double edep0, double edep1,
                                 double edep2, double edep3)
{
    fCoincCount[coinc]++;

    double edeps[4] = {edep0, edep1, edep2, edep3};
    for (int s = 0; s < 4; ++s) {
        if (coinc & (1 << s)) {
            fCoincEdepSum[s] += edeps[s];
            fCoincEdepN[s]++;
        }
    }

    if (coinc == 0xF) fEffNum++;

    for (int x = 0; x < 4; ++x) {
        int trigger = 0xF & ~(1 << x);
        if ((coinc & trigger) == trigger)
            fEffDenom[x]++;
    }
}

void RunAction::EndOfRunAction(const G4Run* run)
{
    fHitsFile.close();
    fCoincFile.close();

    G4int nEvents = run->GetNumberOfEvent();
    if (nEvents == 0) return;

    // ── coincidence_summary.csv ───────────────────────────────────────────
    {
        std::ofstream f("results/coincidence_summary.csv");

        long n_2fold = 0, n_3fold = 0, n_4fold = 0;
        for (int m = 1; m < 16; ++m) {
            int bits = __builtin_popcount(static_cast<unsigned>(m));
            if (bits >= 2) n_2fold += fCoincCount[m];
            if (bits >= 3) n_3fold += fCoincCount[m];
            if (bits == 4) n_4fold += fCoincCount[m];
        }

        f << "# Coincidence summary\n";
        f << "# Threshold: " << kDiscrimThreshold_MeV << " MeV\n";
        f << "# Total events: " << nEvents << "\n#\n";

        f << "class,n_events,rate_per_event\n";
        f << "2fold_any," << n_2fold << ","
          << std::setprecision(6) << double(n_2fold)/nEvents << "\n";
        f << "3fold_any," << n_3fold << ","
          << double(n_3fold)/nEvents << "\n";
        f << "4fold_all," << n_4fold << ","
          << double(n_4fold)/nEvents << "\n";

        // ── Plate efficiencies ────────────────────────────────────────────
        f << "#\n# Plate efficiency (trigger = other 3 fired)\n";
        f << "plate,n_trigger,n_4fold,efficiency,efficiency_err\n";
        for (int x = 0; x < 4; ++x) {
            long denom = fEffDenom[x];
            double eff = denom > 0 ? double(fEffNum)/double(denom) : 0.;
            double err = denom > 0
                       ? std::sqrt(eff * (1.0 - eff) / double(denom))
                       : 0.;
            f << "Scint_" << x << ","
              << denom << "," << fEffNum << ","
              << std::setprecision(6) << eff << "," << err << "\n";
        }

        // ── Mean edep per plate in coincident events ──────────────────────
        f << "#\n# Mean edep per plate when in coincidence\n";
        f << "plate,n_coinc,mean_edep_MeV\n";
        for (int s = 0; s < 4; ++s) {
            double mean = fCoincEdepN[s] > 0
                        ? fCoincEdepSum[s] / fCoincEdepN[s] : 0.;
            f << "Scint_" << s << ","
              << fCoincEdepN[s] << ","
              << std::setprecision(6) << mean << "\n";
        }

        // ── All 15 patterns ──────────────────────────────────────────────
        f << "#\n# All bitmask patterns\n";
        f << "mask,label,n_events,rate_per_event\n";
        for (int m = 1; m < 16; ++m) {
            if (fCoincCount[m] == 0) continue;
            f << m << "," << MaskLabel(m) << ","
              << fCoincCount[m] << ","
              << std::setprecision(6) << double(fCoincCount[m])/nEvents << "\n";
        }

        G4cout << "\n  -> results/coincidence_summary.csv" << G4endl;
    }

    // ── Console summary ───────────────────────────────────────────────────
    long n4 = fCoincCount[0xF];
    G4cout << "\n  4-fold rate: " << double(n4)/nEvents << " per event";
    for (int x = 0; x < 4; ++x) {
        double eff = fEffDenom[x] > 0
                   ? double(fEffNum)/double(fEffDenom[x]) : 0.;
        G4cout << "\n  Scint_" << x << " eff: "
               << std::setprecision(4) << eff
               << "  (n_trig=" << fEffDenom[x] << ")";
    }
    G4cout << "\n\n=== Run complete: " << nEvents << " events ===" << G4endl;
}
