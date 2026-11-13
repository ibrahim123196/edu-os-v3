"""Seed the curriculum graph with NCERT Physics Class 12, chapters 1-4.

Concepts are shared: "Coulomb's law" is one node that CBSE, JEE and NEET all
link to, so a student revising for boards is building mastery the system can
read when they sit JEE.

Topic numbers and names follow the NCERT contents pages. Prerequisites are
the ones the textbook's own ordering implies -- you cannot read Gauss's law
without electric field, and you cannot read electric field without Coulomb's
law. They are deliberately sparse: a wrong prerequisite edge sends Milestone 4
chasing the wrong weakness, which is worse than a missing one.

Idempotent -- re-running updates rather than duplicating.

    python -m scripts.seed_curriculum
"""

from __future__ import annotations

from sqlalchemy import select

from apps.api.curriculum import Chapter, Concept, Exam, Subject, Topic
from apps.api.db import init_db, session_scope

EXAMS = [
    ("cbse-12", "CBSE Class 12", "12"),
    ("jee-main", "JEE Main", "11-12"),
    ("neet", "NEET", "11-12"),
]

# (slug, display name, prerequisite slugs)
CONCEPTS = [
    # Chapter 1 — Electric Charges and Fields
    ("electric-charge", "Electric charge", []),
    ("charge-quantisation", "Quantisation of charge", ["electric-charge"]),
    ("charge-conservation", "Conservation of charge", ["electric-charge"]),
    ("coulombs-law", "Coulomb's law", ["electric-charge"]),
    ("superposition-principle", "Superposition principle", ["coulombs-law"]),
    ("electric-field", "Electric field", ["coulombs-law"]),
    ("field-lines", "Electric field lines", ["electric-field"]),
    ("electric-dipole", "Electric dipole", ["electric-field"]),
    ("dipole-in-field", "Dipole in a uniform external field", ["electric-dipole"]),
    ("electric-flux", "Electric flux", ["electric-field"]),
    ("gauss-law", "Gauss's law", ["electric-flux"]),
    ("gauss-applications", "Applications of Gauss's law", ["gauss-law"]),
    # Chapter 2 — Electrostatic Potential and Capacitance
    ("electrostatic-potential", "Electrostatic potential", ["electric-field"]),
    ("potential-point-charge", "Potential due to a point charge", ["electrostatic-potential"]),
    ("equipotential-surfaces", "Equipotential surfaces", ["electrostatic-potential"]),
    ("potential-energy-charges", "Potential energy of a system of charges", ["electrostatic-potential"]),
    ("conductors-electrostatics", "Conductors in electrostatics", ["electric-field"]),
    ("dielectrics-polarisation", "Dielectrics and polarisation", ["conductors-electrostatics"]),
    ("capacitance", "Capacitance", ["electrostatic-potential", "conductors-electrostatics"]),
    ("parallel-plate-capacitor", "Parallel plate capacitor", ["capacitance"]),
    ("capacitor-combinations", "Capacitors in series and parallel", ["capacitance"]),
    ("energy-in-capacitor", "Energy stored in a capacitor", ["capacitance"]),
    # Chapter 3 — Current Electricity
    ("electric-current", "Electric current", ["electric-charge"]),
    ("drift-velocity", "Drift velocity and mobility", ["electric-current", "electric-field"]),
    ("ohms-law", "Ohm's law", ["electric-current"]),
    ("resistivity", "Resistivity and conductivity", ["ohms-law"]),
    ("temperature-resistivity", "Temperature dependence of resistivity", ["resistivity"]),
    ("electrical-energy-power", "Electrical energy and power", ["ohms-law"]),
    ("resistor-combinations", "Resistors in series and parallel", ["ohms-law"]),
    ("emf-internal-resistance", "EMF and internal resistance", ["ohms-law"]),
    ("kirchhoffs-rules", "Kirchhoff's rules", ["resistor-combinations", "emf-internal-resistance"]),
    ("wheatstone-bridge", "Wheatstone bridge", ["kirchhoffs-rules"]),
    # Chapter 4 — Moving Charges and Magnetism
    ("magnetic-force", "Magnetic force on a moving charge", ["electric-current"]),
    ("lorentz-force", "Lorentz force", ["magnetic-force", "electric-field"]),
    ("motion-in-magnetic-field", "Motion of a charge in a magnetic field", ["lorentz-force"]),
    ("cyclotron", "Cyclotron", ["motion-in-magnetic-field"]),
    ("biot-savart-law", "Biot-Savart law", ["magnetic-force"]),
    ("field-circular-loop", "Magnetic field on the axis of a circular loop", ["biot-savart-law"]),
    ("amperes-circuital-law", "Ampere's circuital law", ["biot-savart-law"]),
    ("solenoid-toroid", "The solenoid and the toroid", ["amperes-circuital-law"]),
    ("force-between-currents", "Force between parallel currents", ["magnetic-force"]),
    ("torque-current-loop", "Torque on a current loop", ["force-between-currents"]),
    ("moving-coil-galvanometer", "Moving coil galvanometer", ["torque-current-loop"]),
]

# chapter number -> (name, [(topic number, topic name, [concept slugs])])
CHAPTERS = {
    1: ("Electric Charges and Fields", [
        ("1.1", "Introduction", ["electric-charge"]),
        ("1.2", "Electric Charge", ["electric-charge"]),
        ("1.3", "Conductors and Insulators", ["electric-charge"]),
        ("1.4", "Basic Properties of Electric Charge", ["charge-quantisation", "charge-conservation"]),
        ("1.5", "Coulomb's Law", ["coulombs-law"]),
        ("1.6", "Forces between Multiple Charges", ["superposition-principle"]),
        ("1.7", "Electric Field", ["electric-field"]),
        ("1.8", "Electric Field Lines", ["field-lines"]),
        ("1.9", "Electric Flux", ["electric-flux"]),
        ("1.10", "Electric Dipole", ["electric-dipole"]),
        ("1.11", "Dipole in a Uniform External Field", ["dipole-in-field"]),
        ("1.12", "Continuous Charge Distribution", ["electric-field"]),
        ("1.13", "Gauss's Law", ["gauss-law"]),
        ("1.14", "Applications of Gauss's Law", ["gauss-applications"]),
    ]),
    2: ("Electrostatic Potential and Capacitance", [
        ("2.1", "Introduction", ["electrostatic-potential"]),
        ("2.2", "Electrostatic Potential", ["electrostatic-potential"]),
        ("2.3", "Potential due to a Point Charge", ["potential-point-charge"]),
        ("2.4", "Potential due to an Electric Dipole", ["potential-point-charge", "electric-dipole"]),
        ("2.5", "Potential due to a System of Charges", ["potential-point-charge"]),
        ("2.6", "Equipotential Surfaces", ["equipotential-surfaces"]),
        ("2.7", "Potential Energy of a System of Charges", ["potential-energy-charges"]),
        ("2.8", "Potential Energy in an External Field", ["potential-energy-charges"]),
        ("2.9", "Electrostatics of Conductors", ["conductors-electrostatics"]),
        ("2.10", "Dielectrics and Polarisation", ["dielectrics-polarisation"]),
        ("2.11", "Capacitors and Capacitance", ["capacitance"]),
        ("2.12", "The Parallel Plate Capacitor", ["parallel-plate-capacitor"]),
        ("2.13", "Effect of Dielectric on Capacitance", ["dielectrics-polarisation", "capacitance"]),
        ("2.14", "Combination of Capacitors", ["capacitor-combinations"]),
        ("2.15", "Energy Stored in a Capacitor", ["energy-in-capacitor"]),
    ]),
    3: ("Current Electricity", [
        ("3.1", "Introduction", ["electric-current"]),
        ("3.2", "Electric Current", ["electric-current"]),
        ("3.3", "Electric Currents in Conductors", ["electric-current"]),
        ("3.4", "Ohm's Law", ["ohms-law"]),
        ("3.5", "Drift of Electrons and the Origin of Resistivity", ["drift-velocity", "resistivity"]),
        ("3.6", "Limitations of Ohm's Law", ["ohms-law"]),
        ("3.7", "Resistivity of Various Materials", ["resistivity"]),
        ("3.8", "Temperature Dependence of Resistivity", ["temperature-resistivity"]),
        ("3.9", "Electrical Energy, Power", ["electrical-energy-power"]),
        # This edition has no separate "Combination of Resistors" section,
        # so the numbering from here on is one lower than the syllabus
        # ordering suggests. Taken from the chapter's own headings rather
        # than from the syllabus, because the textbook is what the passages
        # are tagged against.
        ("3.10", "Cells, EMF, Internal Resistance", ["emf-internal-resistance"]),
        ("3.11", "Cells in Series and in Parallel", ["emf-internal-resistance"]),
        ("3.12", "Kirchhoff's Rules", ["kirchhoffs-rules", "resistor-combinations"]),
        ("3.13", "Wheatstone Bridge", ["wheatstone-bridge"]),
    ]),
    4: ("Moving Charges and Magnetism", [
        ("4.1", "Introduction", ["magnetic-force"]),
        ("4.2", "Magnetic Force", ["magnetic-force", "lorentz-force"]),
        ("4.3", "Motion in a Magnetic Field",
         ["motion-in-magnetic-field", "cyclotron"]),
        ("4.4", "Magnetic Field due to a Current Element, Biot-Savart Law", ["biot-savart-law"]),
        ("4.5", "Magnetic Field on the Axis of a Circular Current Loop", ["field-circular-loop"]),
        ("4.6", "Ampere's Circuital Law", ["amperes-circuital-law"]),
        ("4.7", "The Solenoid", ["solenoid-toroid"]),
        ("4.8", "Force between Two Parallel Currents, the Ampere", ["force-between-currents"]),
        ("4.9", "Torque on Current Loop, Magnetic Dipole", ["torque-current-loop"]),
        ("4.10", "The Moving Coil Galvanometer", ["moving-coil-galvanometer"]),
    ]),
}


def seed() -> None:
    init_db()

    with session_scope() as db:
        # Exams
        exams: dict[str, Exam] = {}
        for code, name, grade in EXAMS:
            exam = db.scalar(select(Exam).where(Exam.code == code))
            if exam is None:
                exam = Exam(code=code, name=name, grade=grade)
                db.add(exam)
            exams[code] = exam

        # Subject
        subject = db.scalar(select(Subject).where(Subject.code == "physics"))
        if subject is None:
            subject = Subject(code="physics", name="Physics")
            db.add(subject)
        db.flush()
        subject.exams = list(exams.values())

        # Concepts, then prerequisite edges once every node exists
        concepts: dict[str, Concept] = {}
        for slug, name, _ in CONCEPTS:
            concept = db.scalar(select(Concept).where(Concept.slug == slug))
            if concept is None:
                concept = Concept(slug=slug, name=name)
                db.add(concept)
            concepts[slug] = concept
        db.flush()

        for slug, _, prerequisite_slugs in CONCEPTS:
            concept = concepts[slug]
            concept.prerequisites = [concepts[p] for p in prerequisite_slugs]
            # Shared: every exam that covers Class 12 physics assesses these.
            concept.exams = list(exams.values())

        # Chapters and topics
        for number, (chapter_name, topics) in CHAPTERS.items():
            chapter = db.scalar(
                select(Chapter).where(
                    Chapter.subject_id == subject.id,
                    Chapter.number == number,
                    Chapter.part == "I",
                )
            )
            if chapter is None:
                chapter = Chapter(
                    subject_id=subject.id, number=number, name=chapter_name, part="I"
                )
                db.add(chapter)
            else:
                chapter.name = chapter_name
            db.flush()

            # Drop topics this seed no longer defines. Without this, editing
            # the numbering leaves orphans behind that no passage can match
            # and that quietly inflate the topic count.
            defined = {number for number, _, _ in topics}
            for existing in list(chapter.topics):
                if existing.number not in defined:
                    print(f"  removing stale topic {existing.number} {existing.name}")
                    db.delete(existing)
            db.flush()

            for topic_number, topic_name, concept_slugs in topics:
                topic = db.scalar(
                    select(Topic).where(
                        Topic.chapter_id == chapter.id, Topic.number == topic_number
                    )
                )
                if topic is None:
                    topic = Topic(
                        chapter_id=chapter.id, number=topic_number, name=topic_name
                    )
                    db.add(topic)
                else:
                    topic.name = topic_name
                db.flush()
                topic.concepts = [concepts[s] for s in concept_slugs]

    with session_scope() as db:
        print(f"exams:     {db.query(Exam).count()}")
        print(f"subjects:  {db.query(Subject).count()}")
        print(f"chapters:  {db.query(Chapter).count()}")
        print(f"topics:    {db.query(Topic).count()}")
        print(f"concepts:  {db.query(Concept).count()}")


if __name__ == "__main__":
    seed()
