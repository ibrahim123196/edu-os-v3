"""Topic groups: what a student means by a topic name.

The course map follows the textbook's sections, so "capacitors" is four
concepts and "Gauss's law" is three. A student asking for a quiz on
capacitance means all four; a quiz drawn from the one concept called
Capacitance once offered 2 questions on a topic that fills a third of a
chapter.

Written out by hand, not derived. Neither concept names nor prerequisites
give the right grouping: by name, Capacitance does not contain "parallel
plate capacitor"; by prerequisite, Electric charge would swallow Coulomb's
law and electric current. A short list someone has read is better than a
rule nobody can check. Agreed with Ibrahim, 17 Nov 2026, including:

- Dielectrics and polarisation stays out of Capacitors, though the textbook
  treats a dielectric's effect on capacitance (2.13): it comes up only when
  the student names it.
- Electric flux belongs with Gauss's law only, not with Electric field.
- Electrical energy and power belongs with Circuits.

Naming a single concept still means just that concept: the longest name
in the message wins, so "parallel plate capacitor" beats "capacitor" and
"applications of Gauss's law" beats "Gauss's law".
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TopicGroup:
    slug: str
    name: str
    members: tuple[str, ...]          # concept slugs, in the textbook's order
    aliases: tuple[str, ...] = ()     # other names a student uses for it
    chapter: bool = False


GROUPS: tuple[TopicGroup, ...] = (
    # --- Chapter 1: Electric charges and fields ---
    TopicGroup("charges", "Electric charge",
               ("electric-charge", "charge-quantisation", "charge-conservation"),
               aliases=("charges",)),
    TopicGroup("coulomb", "Coulomb's law", ("coulombs-law", "superposition-principle")),
    TopicGroup("field", "Electric field", ("electric-field", "field-lines")),
    TopicGroup("dipoles", "Dipoles", ("electric-dipole", "dipole-in-field"),
               aliases=("dipole", "electric dipole", "electric dipoles")),
    TopicGroup("gauss", "Gauss's law", ("electric-flux", "gauss-law", "gauss-applications"),
               aliases=("gauss law", "gauss theorem", "gauss's theorem")),
    # --- Chapter 2: Electrostatic potential and capacitance ---
    TopicGroup("potential", "Electrostatic potential",
               ("electrostatic-potential", "potential-point-charge",
                "equipotential-surfaces", "potential-energy-charges"),
               aliases=("potential",)),
    TopicGroup("capacitors", "Capacitors",
               ("capacitance", "parallel-plate-capacitor", "capacitor-combinations",
                "energy-in-capacitor"),
               aliases=("capacitor", "capacitance")),
    # --- Chapter 3: Current electricity ---
    TopicGroup("current-resistance", "Current and resistance",
               ("electric-current", "drift-velocity", "ohms-law", "resistivity",
                "temperature-resistivity"),
               aliases=("resistance",)),
    TopicGroup("circuits", "Circuits",
               ("resistor-combinations", "emf-internal-resistance", "kirchhoffs-rules",
                "wheatstone-bridge", "electrical-energy-power"),
               aliases=("circuit", "electric circuits")),
    # --- Chapter 4: Moving charges and magnetism ---
    TopicGroup("magnetic-force", "Magnetic force",
               ("magnetic-force", "lorentz-force", "motion-in-magnetic-field", "cyclotron")),
    TopicGroup("field-of-currents", "Magnetic field of currents",
               ("biot-savart-law", "field-circular-loop", "amperes-circuital-law",
                "solenoid-toroid"),
               aliases=("magnetic field of a current", "magnetic field due to current")),
    TopicGroup("forces-on-currents", "Forces and torques on currents",
               ("force-between-currents", "torque-current-loop", "moving-coil-galvanometer")),
    # --- Whole chapters ---
    TopicGroup("ch1", "Electric charges and fields",
               ("electric-charge", "charge-quantisation", "charge-conservation",
                "coulombs-law", "superposition-principle", "electric-field", "field-lines",
                "electric-dipole", "dipole-in-field", "electric-flux", "gauss-law",
                "gauss-applications"),
               aliases=("charges and fields",), chapter=True),
    TopicGroup("ch2", "Electrostatic potential and capacitance",
               ("electrostatic-potential", "potential-point-charge", "equipotential-surfaces",
                "potential-energy-charges", "conductors-electrostatics",
                "dielectrics-polarisation", "capacitance", "parallel-plate-capacitor",
                "capacitor-combinations", "energy-in-capacitor"),
               aliases=("potential and capacitance",), chapter=True),
    TopicGroup("ch3", "Current electricity",
               ("electric-current", "drift-velocity", "ohms-law", "resistivity",
                "temperature-resistivity", "electrical-energy-power", "resistor-combinations",
                "emf-internal-resistance", "kirchhoffs-rules", "wheatstone-bridge"),
               chapter=True),
    TopicGroup("ch4", "Moving charges and magnetism",
               ("magnetic-force", "lorentz-force", "motion-in-magnetic-field", "cyclotron",
                "biot-savart-law", "field-circular-loop", "amperes-circuital-law",
                "solenoid-toroid", "force-between-currents", "torque-current-loop",
                "moving-coil-galvanometer"),
               aliases=("magnetism", "moving charges"), chapter=True),
)

BY_SLUG = {g.slug: g for g in GROUPS}

# Everyday names for single concepts whose full name a student rarely
# types: "dielectrics", not "Dielectrics and polarisation". Without these
# the topic is found only by searching the textbook, which can pick a
# neighbouring section.
CONCEPT_ALIASES: dict[str, tuple[str, ...]] = {
    "dielectrics-polarisation": ("dielectrics", "dielectric", "polarisation", "polarization"),
    "superposition-principle": ("superposition",),
    "drift-velocity": ("drift velocity", "mobility"),
    "resistivity": ("resistivity", "conductivity"),
    "temperature-resistivity": ("temperature dependence of resistance",),
    "emf-internal-resistance": ("emf", "internal resistance", "cells"),
    "kirchhoffs-rules": ("kirchhoff's laws", "kirchhoffs laws", "kirchhoff"),
    "potential-energy-charges": ("potential energy",),
    "equipotential-surfaces": ("equipotentials", "equipotential"),
    "conductors-electrostatics": ("conductors",),
    "field-circular-loop": ("circular loop", "current loop field"),
    "solenoid-toroid": ("solenoid", "toroid"),
    "moving-coil-galvanometer": ("galvanometer",),
    "torque-current-loop": ("torque on a loop",),
    "motion-in-magnetic-field": ("circular motion in a magnetic field",),
}
