//! Atomic data utilities (CODATA / IUPAC 2022-2023)
//!
//! Provides a compile-time table of standard atomic weights (in unified
//! atomic mass units, u) for the 118 recognised chemical elements.
//!
//! References
//! ----------
//! • NIST Standard Reference Database 144 – Atomic Weights and Isotopic Compositions.
//! • IUPAC Technical Report 2021: Standard Atomic Weights of the Elements 2021.
//! • CODATA 2022 fundamental physical constants.
//!
//! The values below use the conventional (interval) standard atomic weights
//! where published. For elements with no stable isotopes the mass of the
//! most stable (longest-lived) isotope is provided following IUPAC
//! convention.

use crate::constants::ATOMIC_MASS_UNIT;

/// Standard atomic weights (u) for Z = 1‥118.
///
/// Index 0 corresponds to Hydrogen (Z=1).
#[rustfmt::skip]
pub const ATOMIC_MASSES_U: [f64; 118] = [
    1.007_84,  // 1  H
    4.002_602, // 2  He
    6.941,     // 3  Li (interval 6.938-6.997 → mean)
    9.012_1831,// 4  Be
   10.81,      // 5  B
   12.011,     // 6  C
   14.007,     // 7  N
   15.999,     // 8  O
   18.998_403_163,// 9  F
   20.179_7,   // 10 Ne
   22.989_769_28, // 11 Na
   24.305,     // 12 Mg
   26.981_538_5, // 13 Al
   28.085,     // 14 Si
   30.973_761_998,//15 P
   32.06,      // 16 S
   35.45,      // 17 Cl
   39.948,     // 18 Ar
   39.0983,    // 19 K
   40.078,     // 20 Ca
   44.955_908, // 21 Sc
   47.867,     // 22 Ti
   50.9415,    // 23 V
   51.996_1,   // 24 Cr
   54.938_044, // 25 Mn
   55.845,     // 26 Fe
   58.933_194, // 27 Co
   58.6934,    // 28 Ni
   63.546,     // 29 Cu
   65.38,      // 30 Zn
   69.723,     // 31 Ga
   72.63,      // 32 Ge
   74.921_595, // 33 As
   78.971,     // 34 Se
   79.904,     // 35 Br
   83.798,     // 36 Kr
   85.4678,    // 37 Rb
   87.62,      // 38 Sr
   88.905_84,  // 39 Y
   91.224,     // 40 Zr
   92.906_37,  // 41 Nb
   95.95,      // 42 Mo
   98.0,       // 43 Tc (long-lived isotope)
  101.07,      // 44 Ru
  102.9055,    // 45 Rh
  106.42,      // 46 Pd
  107.8682,    // 47 Ag
  112.414,     // 48 Cd
  114.818,     // 49 In
  118.71,      // 50 Sn
  121.76,      // 51 Sb
  127.6,       // 52 Te
  126.904_47,  // 53 I
  131.293,     // 54 Xe
  132.905_452, // 55 Cs
  137.327,     // 56 Ba
  138.905_47,  // 57 La
  140.116,     // 58 Ce
  140.907_66,  // 59 Pr
  144.242,     // 60 Nd
  145.0,       // 61 Pm
  150.36,      // 62 Sm
  151.964,     // 63 Eu
  157.25,      // 64 Gd
  158.925_35,  // 65 Tb
  162.5,       // 66 Dy
  164.930_33,  // 67 Ho
  167.259,     // 68 Er
  168.934_22,  // 69 Tm
  173.045,     // 70 Yb
  174.966_8,   // 71 Lu
  178.49,      // 72 Hf
  180.947_88,  // 73 Ta
  183.84,      // 74 W
  186.207,     // 75 Re
  190.23,      // 76 Os
  192.217,     // 77 Ir
  195.084,     // 78 Pt
  196.966_569, // 79 Au
  200.592,     // 80 Hg
  204.38,      // 81 Tl
  207.2,       // 82 Pb
  208.980_4,   // 83 Bi
  209.0,       // 84 Po
  210.0,       // 85 At
  222.0,       // 86 Rn
  223.0,       // 87 Fr
  226.0,       // 88 Ra
  227.0,       // 89 Ac
  232.037_7,   // 90 Th
  231.035_88,  // 91 Pa
  238.028_91,  // 92 U
  237.0,       // 93 Np
  244.0,       // 94 Pu
  243.0,       // 95 Am
  247.0,       // 96 Cm
  247.0,       // 97 Bk
  251.0,       // 98 Cf
  252.0,       // 99 Es
  257.0,       // 100 Fm
  258.0,       // 101 Md
  259.0,       // 102 No
  262.0,       // 103 Lr
  267.0,       // 104 Rf
  270.0,       // 105 Db
  269.0,       // 106 Sg
  270.0,       // 107 Bh
  270.0,       // 108 Hs
  278.0,       // 109 Mt
  281.0,       // 110 Ds
  282.0,       // 111 Rg (approx.)
  285.0,       // 112 Cn
  286.0,       // 113 Nh
  289.0,       // 114 Fl
  290.0,       // 115 Mc
  293.0,       // 116 Lv
  294.0,       // 117 Ts
  294.0,       // 118 Og
];

/// Return atomic mass (kg) for a given atomic number Z (1-118).
pub fn mass_kg(z: u32) -> f64 {
    if z == 0 || z as usize > ATOMIC_MASSES_U.len() {
        return 0.0;
    }
    ATOMIC_MASSES_U[(z - 1) as usize] * ATOMIC_MASS_UNIT
} 