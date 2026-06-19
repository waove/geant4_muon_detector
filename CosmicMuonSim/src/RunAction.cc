#include "RunAction.hh"
#include "G4Run.hh"
#include "G4SystemOfUnits.hh"

#include <filesystem>
#include <iomanip>
#include <cstring>
#include <bitset>

RunAction::RunAction() {}

// ── Helper: bitmask label ─────────────────────────────────────────────────────
static std::string MaskLabel(int mask)
{
    // e.g. mask=0b0101 → "S0+S2"
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

    std::memset(fTotalEdep,     0, sizeof(fTotalEdep));
    std::memset(fTotalStepLen,  0, sizeof(fTotalStepLen));
    std::memset(fTotalEntered,  0, sizeof(fTotalEntered));
    std::memset(fTotalStopped,  0, sizeof(fTotalStopped));
    std::memset(fCondStopped,   0, sizeof(fCondStopped));
    std::memset(fTotalDecays,   0, sizeof(fTotalDecays));
    std::memset(fDecSecEntered, 0, sizeof(fDecSecEntered));
    std::memset(fDecSecStopped, 0, sizeof(fDecSecStopped));
    std::memset(fCoincCount,    0, sizeof(fCoincCount));
    std::memset(fCoincWithAl,   0, sizeof(fCoincWithAl));
    std::memset(fCoincEdepSum,  0, sizeof(fCoincEdepSum));
    std::memset(fCoincEdepN,    0, sizeof(fCoincEdepN));
    std::memset(fEffDenom,      0, sizeof(fEffDenom));
    fEffNum = 0;

    fNDetailedWritten = 0;

    fEdepPerEventFile.open("results/edep_per_event.csv");
    fEdepPerEventFile << "eventID,volume";
    for (int p = 0; p < kNParts; ++p)
        fEdepPerEventFile << "," << kPartNames[p] << "_MeV";
    fEdepPerEventFile << "\n";

    fStepFile.open("results/steps.csv");
    fStepFile << "eventID,plateIdx,trackID,parentID,pdg,particle,"
              << "process,edep_MeV,stepLen_mm,KE_MeV,"
              << "x_mm,y_mm,z_mm,dx,dy,dz,time_ns\n";

    fTrajFile.open("results/trajectories.csv");
    fTrajFile << "eventID,trackID,parentID,pdg,particle,volume,"
              << "edep_MeV,stepLen_mm,KE_MeV,"
              << "x_mm,y_mm,z_mm,dx,dy,dz,time_ns\n";
    G4cout << "Recording detailed trajectories for first "
           << kNDetailEvents << " events" << G4endl;

    fEntryKEFile.open("results/entry_ke.csv");
    fEntryKEFile << "eventID,volIdx,pdg,trackID,ke_MeV\n";

    fDecayFile.open("results/decay.csv");
    fDecayFile << "eventID,volume,pdg,trackID,"
               << "ke_MeV,x_mm,y_mm,z_mm,time_ns\n";

    fDecaySecEntryKEFile.open("results/decay_secondary_entry_ke.csv");
    fDecaySecEntryKEFile << "eventID,volume,pdg,trackID,ke_MeV\n";

    // coincidences.csv — one row per event where ≥1 scintillator fired
    // coinc_mask: 4-bit integer (bit i = Scint_i above threshold)
    // n_fired:    number of plates that fired (0–4)
    // al_fired:   1 if Al plate also above threshold
    // edep_s*:    total energy deposit in each scint [MeV]
    // edep_al:    total energy deposit in Al plate [MeV]
    fCoincFile.open("results/coincidences.csv");
    fCoincFile << "eventID,coinc_mask,n_fired,al_fired,"
               << "edep_s0_MeV,edep_s1_MeV,edep_s2_MeV,edep_s3_MeV,"
               << "edep_al_MeV\n";
    G4cout << "Discriminator threshold: " << kDiscrimThreshold_MeV
           << " MeV per plate" << G4endl;

    fGunEnergyFile.open("results/gun_energy.csv");
    fGunEnergyFile << "eventID,pdg,energy_MeV,cosTheta\n";
}

void RunAction::AccumulateEdep(int v, int p, double edep)
    { fTotalEdep[v][p] += edep; }

void RunAction::AccumulateStepLen(int v, int p, double stepLen)
    { fTotalStepLen[v][p] += stepLen; }

void RunAction::AccumulateTracking(int v, int p, int nEntered, int nStopped)
    { fTotalEntered[v][p] += nEntered; fTotalStopped[v][p] += nStopped; }

void RunAction::AccumulateConditionalStopping(int vol, int part, int nStopped)
    { fCondStopped[vol][part] += nStopped; }

void RunAction::AccumulateDecay(int vol, int muonIdx)
{
    if (muonIdx >= 0 && muonIdx < 2)
        fTotalDecays[vol][muonIdx]++;
}

void RunAction::AccumulateDecaySec(int vol, int part, int nEntered, int nStopped)
{
    fDecSecEntered[vol][part] += nEntered;
    fDecSecStopped[vol][part] += nStopped;
}

void RunAction::AccumulateCoinc(int coinc, bool alFired,
                                 double edep0, double edep1,
                                 double edep2, double edep3,
                                 double edepAl)
{
    // Count this bitmask pattern
    fCoincCount[coinc]++;
    if (alFired) fCoincWithAl[coinc]++;

    // Accumulate edep for plates that fired
    double edeps[4] = {edep0, edep1, edep2, edep3};
    for (int s = 0; s < 4; ++s) {
        if (coinc & (1 << s)) {
            fCoincEdepSum[s] += edeps[s];
            fCoincEdepN  [s]++;
        }
    }

    // Efficiency counters
    // 4-fold: all four fired
    if (coinc == 0xF) {
        fEffNum++;
    }
    // 3-fold trigger for each plate's efficiency:
    // plate x efficiency denominator = events where all plates EXCEPT x fired
    for (int x = 0; x < 4; ++x) {
        int trigger_mask = 0xF & ~(1 << x);  // all bits except x
        // The three trigger plates all fired (regardless of plate x)
        if ((coinc & trigger_mask) == trigger_mask)
            fEffDenom[x]++;
    }
}

void RunAction::EndOfRunAction(const G4Run* run)
{
    fEdepPerEventFile.close();
    fStepFile.close();
    fTrajFile.close();
    fEntryKEFile.close();
    fDecayFile.close();
    fDecaySecEntryKEFile.close();
    fCoincFile.close();
    fGunEnergyFile.close();

    G4int nEvents = run->GetNumberOfEvent();
    if (nEvents == 0) return;

    {
        std::ofstream f("results/edep_summary.csv");
        f << "volume,particle,total_edep_MeV,mean_edep_MeV\n";
        for (int v = 0; v < kNVols; ++v)
            for (int p = 0; p < kNParts; ++p)
                f << kVolNames[v] << "," << kPartNames[p] << ","
                  << std::setprecision(6)
                  << fTotalEdep[v][p]/MeV << ","
                  << fTotalEdep[v][p]/MeV/nEvents << "\n";
        G4cout << "\n  -> results/edep_summary.csv" << G4endl;
    }
    {
        std::ofstream f("results/stopping.csv");
        f << "volume,particle,n_entered,n_stopped,"
             "stop_frac,stop_frac_vs_scint0\n";
        for (int v = 0; v < kNVols; ++v)
            for (int p = 0; p < kNParts; ++p) {
                long ent = fTotalEntered[v][p], stp = fTotalStopped[v][p];
                double frac    = ent > 0 ? double(stp)/double(ent) : 0.;
                long   ref     = fTotalEntered[kScint0][p];
                double fracRef = ref > 0 ? double(stp)/double(ref) : 0.;
                f << kVolNames[v] << "," << kPartNames[p] << ","
                  << ent << "," << stp << ","
                  << std::setprecision(6) << frac << "," << fracRef << "\n";
            }
        G4cout << "  -> results/stopping.csv" << G4endl;
    }
    {
        std::ofstream f("results/stopping_conditional.csv");
        f << "volume,particle,n_scint0_entered,n_stopped_here,"
             "cond_stop_frac\n";
        for (int v = 0; v < kNVols; ++v)
            for (int p = 0; p < kNParts; ++p) {
                long ref = fTotalEntered[kScint0][p];
                long stp = fCondStopped[v][p];
                double frac = ref > 0 ? double(stp)/double(ref) : 0.;
                f << kVolNames[v] << "," << kPartNames[p] << ","
                  << ref << "," << stp << ","
                  << std::setprecision(6) << frac << "\n";
            }
        G4cout << "  -> results/stopping_conditional.csv" << G4endl;
    }
    {
        std::ofstream f("results/fluence.csv");
        f << "volume,particle,total_steplen_mm,volume_mm3,"
          << "fluence_per_mm2,fluence_per_event_per_mm2\n";
        for (int v = 0; v < kNVols; ++v)
            for (int p = 0; p < kNParts; ++p) {
                double vol = kVolumeMM3[v];
                double fl  = vol > 0 ? fTotalStepLen[v][p]/vol : 0.;
                f << kVolNames[v] << "," << kPartNames[p] << ","
                  << std::setprecision(6)
                  << fTotalStepLen[v][p]/mm << "," << vol << ","
                  << fl << "," << fl/nEvents << "\n";
            }
        G4cout << "  -> results/fluence.csv" << G4endl;
    }
    {
        std::ofstream f("results/decay_summary.csv");
        f << "volume,n_decay_mum,n_decay_mup,n_decay_total,"
             "decay_rate_mum_per_event,decay_rate_mup_per_event\n";
        long grand_mum = 0, grand_mup = 0;
        for (int v = 0; v < kNVols; ++v) {
            long nm = fTotalDecays[v][0], np = fTotalDecays[v][1];
            grand_mum += nm; grand_mup += np;
            if (nm + np == 0) continue;
            f << kVolNames[v] << ","
              << nm << "," << np << "," << nm + np << ","
              << std::setprecision(6)
              << double(nm)/nEvents << "," << double(np)/nEvents << "\n";
        }
        f << "TOTAL,"
          << grand_mum << "," << grand_mup << ","
          << grand_mum + grand_mup << ","
          << double(grand_mum)/nEvents << ","
          << double(grand_mup)/nEvents << "\n";
        G4cout << "  -> results/decay_summary.csv" << G4endl;
    }
    {
        std::ofstream f("results/decay_secondary_stopping.csv");
        f << "volume,particle,n_entered,n_stopped,"
             "stop_frac,stop_frac_vs_total\n";
        long total_sec = 0;
        for (int v = 0; v < kNVols; ++v)
            total_sec += fDecSecEntered[v][kAll];
        for (int v = 0; v < kNVols; ++v)
            for (int p = 0; p < kNParts; ++p) {
                long ent = fDecSecEntered[v][p], stp = fDecSecStopped[v][p];
                if (ent == 0 && stp == 0) continue;
                double frac    = ent > 0 ? double(stp)/double(ent) : 0.;
                double fracTot = double(stp)/double(total_sec > 0 ? total_sec : 1);
                f << kVolNames[v] << "," << kPartNames[p] << ","
                  << ent << "," << stp << ","
                  << std::setprecision(6) << frac << "," << fracTot << "\n";
            }
        G4cout << "  -> results/decay_secondary_stopping.csv" << G4endl;
    }

    // ==============================================================
    // coincidence_summary.csv
    //
    // Named coincidence classes used in the real experiment:
    //
    //   n_2fold_any   — at least 2 plates fired (any pair)
    //   n_3fold_any   — at least 3 plates fired (any triple)
    //   n_4fold       — all 4 plates fired (through-going muon trigger)
    //
    // For each named pattern, the rate per event and the mean edep
    // in each plate (when it was part of that coincidence) are given.
    //
    // Efficiency of plate X:
    //   eff_X = n_4fold / n_3fold_excl_X
    //   where n_3fold_excl_X = events where all plates EXCEPT X fired.
    //   This mirrors the real measurement: use the other 3 plates as
    //   the trigger and ask how often plate X also fires.
    //
    // All 15 non-zero bitmask patterns are also listed individually.
    // ==============================================================
    {
        std::ofstream f("results/coincidence_summary.csv");

        // ── Named classes ────────────────────────────────────────────────
        long n_1fold = 0, n_2fold = 0, n_3fold = 0, n_4fold = 0;
        for (int m = 1; m < 16; ++m) {
            int bits = __builtin_popcount(static_cast<unsigned>(m));
            long cnt = fCoincCount[m];
            if (bits == 1) n_1fold += cnt;
            if (bits >= 2) n_2fold += cnt;
            if (bits >= 3) n_3fold += cnt;
            if (bits == 4) n_4fold += cnt;
        }
        // n_2fold_any includes events with 3-fold and 4-fold
        // To get "exactly 2" use bits==2, but "at least 2" matches
        // real coincidence trigger behaviour.

        f << "# Coincidence summary\n";
        f << "# Discriminator threshold: " << kDiscrimThreshold_MeV
          << " MeV per plate\n";
        f << "# Total events: " << nEvents << "\n";
        f << "#\n";

        f << "class,n_events,rate_per_event\n";
        f << "1fold_any,"  << n_1fold << "," << std::setprecision(6)
          << double(n_1fold)/nEvents << "\n";
        f << "2fold_any,"  << n_2fold << "," << double(n_2fold)/nEvents << "\n";
        f << "3fold_any,"  << n_3fold << "," << double(n_3fold)/nEvents << "\n";
        f << "4fold_all,"  << n_4fold << "," << double(n_4fold)/nEvents << "\n";

        // Pair coincidences (like the real plateau+coincidence measurement)
        f << "#\n# 2-fold pairs\n";
        f << "pair,n_events,rate_per_event\n";
        const char* pair_names[6] = {
            "S0+S1","S0+S2","S0+S3","S1+S2","S1+S3","S2+S3"};
        int pair_masks[6] = {0b0011,0b0101,0b1001,0b0110,0b1010,0b1100};
        for (int i = 0; i < 6; ++i) {
            // Sum over all masks that contain this pair
            long cnt = 0;
            for (int m = 1; m < 16; ++m)
                if ((m & pair_masks[i]) == pair_masks[i])
                    cnt += fCoincCount[m];
            f << pair_names[i] << "," << cnt << ","
              << double(cnt)/nEvents << "\n";
        }

        // ── Efficiency ───────────────────────────────────────────────────
        f << "#\n# Plate efficiency  (trigger = other 3 plates fired)\n";
        f << "plate,n_trigger,n_4fold,efficiency,efficiency_err\n";
        for (int x = 0; x < 4; ++x) {
            long denom = fEffDenom[x];
            long num   = fEffNum;   // 4-fold is the numerator for all plates
            double eff = denom > 0 ? double(num)/double(denom) : 0.;
            // Binomial error: σ = sqrt(eff*(1-eff)/n)
            double err = denom > 0
                       ? std::sqrt(eff * (1.0 - eff) / double(denom))
                       : 0.;
            f << "Scint_" << x << ","
              << denom << "," << num << ","
              << std::setprecision(6) << eff << ","
              << err << "\n";
        }

        // ── Mean edep per plate in coincident events ──────────────────────
        f << "#\n# Mean edep per plate when that plate was in a coincidence\n";
        f << "plate,n_coinc_events,mean_edep_MeV\n";
        for (int s = 0; s < 4; ++s) {
            double mean = fCoincEdepN[s] > 0
                        ? fCoincEdepSum[s] / fCoincEdepN[s]
                        : 0.;
            f << "Scint_" << s << ","
              << fCoincEdepN[s] << ","
              << std::setprecision(6) << mean << "\n";
        }

        // ── All 15 bitmask patterns ───────────────────────────────────────
        f << "#\n# All coincidence patterns (bitmask, label, count, rate)\n";
        f << "mask,label,n_events,rate_per_event,n_with_al\n";
        for (int m = 1; m < 16; ++m) {
            if (fCoincCount[m] == 0) continue;
            f << m << "," << MaskLabel(m) << ","
              << fCoincCount[m] << ","
              << std::setprecision(6) << double(fCoincCount[m])/nEvents << ","
              << fCoincWithAl[m] << "\n";
        }

        G4cout << "  -> results/coincidence_summary.csv" << G4endl;

        // Console summary
        G4cout << "\n  Coincidence rates per event:"      << G4endl;
        G4cout << "    1-fold (any):  " << double(n_1fold)/nEvents << G4endl;
        G4cout << "    2-fold (any):  " << double(n_2fold)/nEvents << G4endl;
        G4cout << "    3-fold (any):  " << double(n_3fold)/nEvents << G4endl;
        G4cout << "    4-fold (all):  " << double(n_4fold)/nEvents << G4endl;
        G4cout << "\n  Plate efficiencies:" << G4endl;
        for (int x = 0; x < 4; ++x) {
            double eff = fEffDenom[x] > 0
                       ? double(fEffNum)/double(fEffDenom[x])
                       : 0.;
            G4cout << "    Scint_" << x << ":  "
                   << std::setprecision(4) << eff
                   << "  (trigger n=" << fEffDenom[x] << ")" << G4endl;
        }
    }

    G4cout << "\n=== Run complete: " << nEvents << " events ===" << G4endl;
}
