"""Build 2,500 diverse topics for a future 50,000-probe compact dataset."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT = Path("topics/50k_rows/2500_topics.json")
DEFAULT_REPORT = Path("topics/50k_rows/coverage_report.json")
EXPECTED_TOPICS = 2_500
PROBES_PER_CHAT = 20
EXPECTED_PROBES = EXPECTED_TOPICS * PROBES_PER_CHAT
INSPIRATION_PATHS = (
    Path("topics/100k/100k_topics.json"),
    Path("topics/500k/500k_topics.json"),
    Path("topics/1M/1M_topics.json"),
    Path("topics/10M/10M_topics.json"),
)


def _projects(value: str) -> list[tuple[str, str]]:
    result = []
    for line in value.strip().splitlines():
        title, goal = line.strip().split(" :: ", maxsplit=1)
        result.append((title, goal))
    return result


def _focus(value: str) -> list[str]:
    return [item.strip() for item in value.split("|") if item.strip()]


SCENARIO_VARIANTS = (
    {
        "label": "Beginner Roadmap",
        "theme": "Start from first principles, establish a baseline, and build confidence through small milestones.",
        "angles": ("Beginner-friendly explanations", "Progressive milestone planning"),
    },
    {
        "label": "Evolving Requirements",
        "theme": "Revisit the work over several sessions as requirements, constraints, and priorities change.",
        "angles": ("Requirement change tracking", "Updating earlier decisions"),
    },
    {
        "label": "Troubleshooting and Correction",
        "theme": "Diagnose setbacks, compare conflicting observations, and document the final resolution.",
        "angles": ("Root-cause investigation", "Contradiction resolution"),
    },
    {
        "label": "Options and Trade-offs",
        "theme": "Compare multiple approaches while retaining the user's stated preferences and decision criteria.",
        "angles": ("Alternative comparison", "Preference-aware recommendations"),
    },
    {
        "label": "Budget and Resource Limits",
        "theme": "Plan under explicit cost, time, staffing, equipment, or compute constraints that later evolve.",
        "angles": ("Resource allocation", "Budget update handling"),
    },
    {
        "label": "Accessibility and Inclusion",
        "theme": "Adapt the plan for different abilities, languages, schedules, backgrounds, and access needs.",
        "angles": ("Accessibility requirements", "Inclusive alternative design"),
    },
    {
        "label": "Privacy Safety and Reliability",
        "theme": "Identify risks, set protective boundaries, and verify the plan before consequential action.",
        "angles": ("Risk and privacy review", "Safety and reliability checks"),
    },
    {
        "label": "Collaboration and Handoff",
        "theme": "Coordinate named participants, responsibilities, reviews, approvals, and a clear handoff.",
        "angles": ("Role and responsibility tracking", "Documentation and handoff"),
    },
    {
        "label": "Feedback and Iteration",
        "theme": "Use measurements and stakeholder feedback to revise the approach across multiple sessions.",
        "angles": ("Success metric selection", "Feedback-driven iteration"),
    },
    {
        "label": "Deadline and Contingency",
        "theme": "Work toward dated milestones while responding to delays, dependencies, and backup plans.",
        "angles": ("Timeline and dependency management", "Contingency planning"),
    },
)


DOMAIN_SPECS = {
    "Software Development": {
        "family": "Technical",
        "projects": _projects(
            """
Building a Personal Budgeting Web Application :: Design and deliver a maintainable web app for tracking income, expenses, and goals
Creating a Cross-Platform Habit Tracker :: Build a small mobile-first application that helps users establish and review habits
Integrating Third-Party APIs into a Service :: Connect external APIs reliably while handling authentication, errors, and changing schemas
Debugging a Flaky Automated Test Suite :: Find nondeterministic failures and restore confidence in continuous integration
Refactoring a Legacy Monolith :: Improve structure and maintainability without breaking existing behavior
Designing a Database for a Growing Product :: Model evolving data requirements and improve query performance safely
Building a Command-Line Automation Tool :: Turn a repetitive workflow into a documented and testable CLI utility
Making a Frontend Accessible :: Audit and improve keyboard, screen-reader, contrast, and responsive behavior
Creating a Real-Time Collaboration Feature :: Add synchronized updates, presence, and conflict handling to an application
Preparing an Open-Source Contribution :: Understand a codebase, propose a change, write tests, and navigate review
"""
        ),
        "focus": _focus(
            "requirements and acceptance criteria|architecture and component boundaries|implementation sequencing|version control workflow|testing strategy|debugging evidence|data modeling|API contracts|performance measurement|security fundamentals|deployment and rollback|documentation and maintenance"
        ),
    },
    "Data Science and Analytics": {
        "family": "Technical",
        "projects": _projects(
            """
Cleaning a Messy Survey Dataset :: Transform inconsistent survey responses into an auditable analysis-ready table
Building an Executive Sales Dashboard :: Define trustworthy metrics and present changes in business performance clearly
Exploring Customer Churn Patterns :: Investigate retention signals without overstating causal conclusions
Planning and Interpreting an A B Test :: Design a controlled experiment and communicate uncertainty in the results
Forecasting Product Demand :: Create a time-series baseline and update it as seasonality and constraints change
Classifying Customer Support Requests :: Organize incoming requests into useful categories with measurable quality
Detecting Anomalies in Sensor Data :: Separate real operational problems from noise and faulty instrumentation
Creating a Reusable SQL Reporting Layer :: Consolidate repeated queries into consistent documented reporting logic
Turning Analysis into a Data Story :: Choose visualizations and narrative structure that communicate findings accurately
Building a Reproducible Research Pipeline :: Track data versions, transformations, assumptions, and generated artifacts
"""
        ),
        "focus": _focus(
            "data provenance|schema inspection|missing-data handling|quality validation|exploratory analysis|metric definitions|statistical assumptions|visualization choices|bias and representativeness|model evaluation|reproducible workflows|communicating uncertainty"
        ),
    },
    "IT Cloud and DevOps": {
        "family": "Technical",
        "projects": _projects(
            """
Improving a Home Office Network :: Diagnose connectivity problems and create a stable secure network setup
Repairing a Continuous Delivery Pipeline :: Trace build and deployment failures across tools and environments
Containerizing a Small Application :: Package an application consistently for development testing and deployment
Planning a Nonprofit Cloud Migration :: Move services gradually while controlling cost downtime and operational risk
Designing Service Monitoring and Alerts :: Select meaningful signals and reduce noisy or missed incidents
Coordinating a Production Incident :: Maintain a timeline assign roles communicate status and capture lessons
Creating a Backup and Recovery Plan :: Define recovery objectives test restores and close coverage gaps
Reviewing Identity and Access Management :: Align account permissions with roles and reduce unnecessary access
Reducing Cloud Infrastructure Costs :: Find waste compare trade-offs and preserve reliability while optimizing spend
Operating a Small Kubernetes Deployment :: Establish practical deployment scaling observability and recovery practices
"""
        ),
        "focus": _focus(
            "environment inventory|configuration management|network diagnostics|deployment pipelines|observability and alerting|incident timelines|backup verification|access controls|capacity planning|cost attribution|rollback procedures|runbooks and ownership"
        ),
    },
    "Cybersecurity and Privacy": {
        "family": "Technical",
        "projects": _projects(
            """
Rolling Out a Password Manager :: Help a small team adopt stronger credential practices and recovery procedures
Creating a Phishing Awareness Program :: Teach recognition reporting and follow-up through realistic defensive examples
Threat Modeling a Web Application :: Identify assets trust boundaries abuse cases and practical mitigations
Hardening a Home Wi-Fi Setup :: Improve router device and guest-network security without unnecessary complexity
Reviewing Privacy for a New SaaS Tool :: Map collected data permissions retention and vendor commitments
Investigating Suspicious Account Activity :: Organize logs and observations for a defensive incident review
Prioritizing Vulnerability Reports :: Triage findings by evidence exposure impact and remediation difficulty
Building a Secure Coding Checklist :: Integrate practical security checks into development and review workflows
Preparing for Ransomware Recovery :: Strengthen backups isolation response roles and business continuity
Completing a Vendor Security Questionnaire :: Gather accurate evidence and distinguish verified controls from assumptions
"""
        ),
        "focus": _focus(
            "asset inventory|threat modeling|authentication and recovery|least privilege|data minimization|secure configuration|logging and detection|vulnerability triage|incident containment|backup resilience|vendor risk|defensive documentation"
        ),
    },
    "Mathematics and Statistics": {
        "family": "Technical",
        "projects": _projects(
            """
Rebuilding Algebra Fundamentals :: Connect symbolic manipulation equations functions and word problems step by step
Exploring Triangle Geometry :: Use diagrams theorems and proofs to reason about geometric relationships
Learning Introductory Calculus :: Develop intuition for limits derivatives integrals and applications
Understanding Probability through Games :: Analyze chance dependence counting and common probability misconceptions
Designing a Statistical Study :: Match questions variables samples and analyses while respecting uncertainty
Applying Linear Algebra to Data :: Understand vectors matrices transformations and simple computational applications
Solving Discrete Mathematics Problems :: Work with logic combinatorics graphs recurrences and proof techniques
Optimizing a Constrained Plan :: Translate a practical allocation problem into objectives constraints and trade-offs
Using Financial Mathematics for Planning :: Explore interest inflation amortization and scenario comparisons educationally
Training for Mathematical Problem Solving :: Build a repeatable approach to unfamiliar multi-step problems and proofs
"""
        ),
        "focus": _focus(
            "definitions and notation|worked examples|visual intuition|assumption checking|symbolic derivation|numerical verification|proof structure|counterexamples|error analysis|real-world modeling|alternative solution methods|practice progression"
        ),
    },
    "Physical Sciences and Engineering": {
        "family": "Science",
        "projects": _projects(
            """
Analyzing Household Energy Use :: Apply physical reasoning to understand energy consumption efficiency and trade-offs
Designing a Mechanics Laboratory :: Plan measurements for motion forces energy and experimental uncertainty
Building and Testing Simple Circuits :: Develop a safe low-voltage electronics project from schematic to measurement
Planning an Astronomy Observation Night :: Select targets equipment timing and fallback conditions for observations
Designing a Safe Chemistry Demonstration :: Explain reactions measurements and laboratory precautions for supervised learning
Comparing Materials for a Prototype :: Evaluate strength weight durability manufacturability and environmental impact
Investigating Heat Transfer :: Model conduction convection radiation and measurement in a practical system
Planning a Small Renewable Energy System :: Estimate needs compare technologies and document engineering assumptions
Prototyping an Environmental Sensor :: Integrate sensors calibration data logging enclosures and field testing
Quantifying Measurement Uncertainty :: Build an uncertainty budget and improve the reliability of reported results
"""
        ),
        "focus": _focus(
            "physical principles|engineering requirements|units and dimensional analysis|experimental design|instrument selection|calibration|measurement uncertainty|safety controls|model validation|materials and components|failure analysis|technical reporting"
        ),
    },
    "Biology and Environmental Science": {
        "family": "Science",
        "projects": _projects(
            """
Studying Ecology in a Community Garden :: Observe interactions among plants pollinators pests soil and seasonal change
Running a Local Biodiversity Survey :: Design repeatable observations and document species evidence responsibly
Monitoring Stream Water Quality :: Plan safe sampling track indicators and communicate environmental limitations
Learning Genetics with Family-Neutral Examples :: Explore inheritance variation and probability without inferring personal diagnoses
Designing a Safe Microbiology Classroom Study :: Use nonpathogenic examples controls and careful observation practices
Creating a Habitat Conservation Plan :: Balance species needs land use community input and measurable outcomes
Interpreting Climate Data :: Compare trends variability baselines and uncertainty across multiple evidence sources
Improving a Composting System :: Understand decomposition moisture aeration temperature and material balance
Organizing a Pollinator Citizen-Science Project :: Coordinate observations identification quality checks and seasonal reporting
Writing a Biological Field Report :: Turn observations methods results and limitations into a clear scientific account
"""
        ),
        "focus": _focus(
            "biological mechanisms|ecosystem relationships|observation protocols|sampling design|controls and comparison|species identification|data quality|seasonal effects|ethical field practices|biosafety boundaries|evidence interpretation|scientific communication"
        ),
    },
    "Health and Wellness": {
        "family": "Personal and Learning",
        "projects": _projects(
            """
Building a Sustainable Exercise Routine :: Create a gradual activity plan around preferences schedule and professional guidance
Improving Sleep Habits :: Track routines and environmental factors while avoiding unsupported medical conclusions
Preparing for a Medical Appointment :: Organize symptoms timelines questions medications and records for a clinician
Maintaining a Symptom Journal :: Record observations consistently without using the chatbot to diagnose a condition
Planning Balanced Everyday Meals :: Explore practical food variety preferences and established dietary guidance
Creating a Stress-Management Routine :: Combine low-risk coping practices boundaries reflection and support resources
Following a Physical Therapy Home Plan :: Organize reminders progress notes and questions around clinician-provided exercises
Organizing Medication Questions :: Prepare an accurate list for a pharmacist or clinician without changing treatment independently
Improving Workplace Ergonomics :: Adjust tasks posture equipment and break patterns based on comfort and guidance
Supporting a Caregiver's Well-Being :: Plan respite communication task sharing and access to qualified support
"""
        ),
        "focus": _focus(
            "goal and baseline tracking|daily routines|preferences and barriers|gradual progression|warning signs and escalation|professional guidance|record accuracy|habit sustainability|environmental adjustments|support networks|privacy of health information|review and adaptation"
        ),
    },
    "Education and Tutoring": {
        "family": "Personal and Learning",
        "projects": _projects(
            """
Creating a Semester Study Plan :: Balance courses assignments review and rest across an evolving academic calendar
Designing an Engaging Lesson :: Connect learning objectives activities checks for understanding and reflection
Preparing for a Cumulative Exam :: Diagnose knowledge gaps schedule retrieval practice and revise priorities
Explaining a Difficult Concept Multiple Ways :: Use analogies examples diagrams and questions suited to the learner
Planning a Home Learning Project :: Coordinate age-appropriate research making presentation and assessment
Strengthening Reading Comprehension :: Practice prediction annotation questioning summarization and evidence use
Building a Spaced-Repetition System :: Turn course material into useful prompts and an adaptive review schedule
Planning a Safe Classroom Investigation :: Align inquiry materials controls observations and supervision
Creating a Fair Grading Rubric :: Define transparent criteria performance levels examples and feedback loops
Supporting Accessible Learning :: Adapt materials pacing interaction and assessment to documented learner needs
"""
        ),
        "focus": _focus(
            "learning objectives|prior knowledge|scaffolding|worked examples|retrieval practice|formative assessment|feedback quality|motivation and pacing|accessibility accommodations|academic integrity|progress monitoring|reflection and transfer"
        ),
    },
    "Writing and Editing": {
        "family": "Personal and Learning",
        "projects": _projects(
            """
Creating a Targeted Resume :: Present relevant experience clearly for a specific role and applicant tracking workflow
Drafting a Persuasive Cover Letter :: Connect evidence from experience to an employer's stated needs
Developing an Academic Essay :: Move from research question to defensible thesis evidence counterargument and revision
Writing Technical Documentation :: Help readers install understand operate and troubleshoot a system
Planning a Long-Form Fiction Project :: Develop characters setting conflict continuity scenes and revision milestones
Preparing a Grant Proposal :: Align a documented need activities outcomes budget and evaluation plan
Improving Important Professional Emails :: Calibrate purpose tone context requests and follow-up for different audiences
Producing a Decision Report :: Synthesize evidence options risks recommendations and unresolved questions
Building a Sustainable Blog Workflow :: Plan audience themes research drafting editing publishing and maintenance
Creating a Personal Editing System :: Use layered revision checklists feedback and version tracking to improve over time
"""
        ),
        "focus": _focus(
            "audience and purpose|structure and outlining|evidence and examples|tone and voice|clarity and concision|argument and logic|continuity and consistency|fact checking|citation and attribution|revision strategy|feedback integration|final quality checks"
        ),
    },
    "Language Learning and Translation": {
        "family": "Personal and Learning",
        "projects": _projects(
            """
Building a Conversation Practice Routine :: Develop confidence through recurring real-life speaking situations and feedback
Learning High-Value Vocabulary :: Organize words by context retrieval cues collocations and spaced review
Understanding a Difficult Grammar Pattern :: Compare form meaning context exceptions and common learner errors
Translating a Community Information Guide :: Preserve meaning tone accessibility and locally appropriate terminology
Preparing Useful Phrases for Travel :: Practice flexible communication for transport lodging food and unexpected changes
Improving Professional Writing in a Second Language :: Revise messages reports and requests for clarity and cultural fit
Working on Pronunciation and Listening :: Use sound contrasts rhythm shadowing transcripts and self-observation
Reading a Short Story in the Target Language :: Build comprehension through context vocabulary discussion and retelling
Creating a Bilingual Project Glossary :: Maintain consistent terminology definitions examples and reviewer decisions
Localizing Product Interface Text :: Adapt concise UI language while respecting space tone and user expectations
"""
        ),
        "focus": _focus(
            "communicative goals|vocabulary in context|grammar and usage|pronunciation cues|listening comprehension|reading strategies|writing feedback|translation accuracy|terminology consistency|cultural context|register and tone|practice and review"
        ),
    },
    "Research and Knowledge Work": {
        "family": "Personal and Learning",
        "projects": _projects(
            """
Planning a Literature Review :: Define scope search systematically compare sources and identify defensible gaps
Refining a Research Question :: Move from a broad interest to a feasible precise and ethically answerable question
Building a Personal Knowledge System :: Capture connect retrieve and review notes without losing source context
Evaluating Conflicting Sources :: Compare authority methods dates evidence incentives and uncertainty
Designing an Interview Protocol :: Create neutral questions consent procedures probes and a consistent workflow
Drafting a Survey Instrument :: Align questions with constructs reduce ambiguity and plan a small pilot
Organizing Qualitative Coding :: Develop a codebook record decisions and compare interpretations transparently
Synthesizing Multiple Reports :: Reconcile definitions time periods findings and limitations across documents
Managing Citations and Research Files :: Establish naming metadata backup deduplication and reference practices
Making an Analysis Reproducible :: Document inputs transformations parameters outputs and known limitations
"""
        ),
        "focus": _focus(
            "scope definition|search strategy|source credibility|evidence extraction|note provenance|method selection|ethics and consent|bias and limitations|conflicting findings|synthesis structure|citation management|reproducibility"
        ),
    },
    "Career and Professional Development": {
        "family": "Professional",
        "projects": _projects(
            """
Organizing a Focused Job Search :: Define target roles track applications tailor materials and learn from results
Practicing for Behavioral Interviews :: Build truthful evidence-rich stories and adapt them to different competencies
Preparing a Promotion Case :: Document impact growth feedback scope and readiness for a larger role
Closing a Professional Skill Gap :: Choose learning projects practice feedback and evidence of competence
Building a Career Portfolio :: Select artifacts explain decisions protect confidential details and show progression
Preparing for a Performance Review :: Summarize outcomes challenges feedback goals and support requests
Planning a Career Change :: Explore transferable skills experiments constraints and staged transitions
Developing a Genuine Networking Routine :: Build reciprocal professional relationships through useful consistent contact
Preparing for a Compensation Conversation :: Research ranges organize evidence and plan respectful negotiation scenarios
Creating a First Ninety-Day Plan :: Learn stakeholders systems priorities risks and early deliverables in a new role
"""
        ),
        "focus": _focus(
            "goal clarification|skills inventory|evidence of impact|market research|application tracking|interview practice|feedback integration|relationship building|negotiation preparation|transition risks|milestone planning|reflection and adjustment"
        ),
    },
    "Business and Entrepreneurship": {
        "family": "Professional",
        "projects": _projects(
            """
Validating a Small Business Idea :: Test the problem audience alternatives and willingness to adopt before overinvesting
Writing a Practical Business Plan :: Connect customer needs operations resources finances risks and milestones
Designing a Pricing Experiment :: Compare value costs positioning and customer response with clear guardrails
Improving a Customer Support Workflow :: Define intake prioritization responses escalation feedback and knowledge reuse
Documenting a Repeatable Operating Procedure :: Capture roles steps exceptions evidence and ownership for a key process
Planning Inventory for a Small Retailer :: Balance demand lead times storage cash constraints and stockout risk
Comparing Potential Vendors :: Evaluate fit reliability terms support risk and total cost consistently
Building a Basic Financial Forecast :: Explore assumptions cash timing scenarios and uncertainty for planning purposes
Planning the First Team Hire :: Define outcomes role scope selection process onboarding and review checkpoints
Coordinating a Product Launch :: Align product operations communications support metrics and contingency plans
"""
        ),
        "focus": _focus(
            "customer problem definition|stakeholder assumptions|market alternatives|value proposition|operating process|resource planning|pricing and costs|vendor management|risk register|success metrics|launch sequencing|post-launch learning"
        ),
    },
    "Personal Finance and Consumer Decisions": {
        "family": "Professional",
        "projects": _projects(
            """
Creating a Household Budget :: Organize income recurring costs variable spending savings and periodic reviews
Comparing Debt Repayment Scenarios :: Understand balances rates minimums trade-offs and questions for qualified advisers
Building an Emergency Fund Plan :: Set a practical target contribution schedule storage criteria and review triggers
Auditing Recurring Subscriptions :: Find low-value renewals cancellation constraints and better household defaults
Planning for a Major Purchase :: Compare total cost timing alternatives financing questions and opportunity costs
Reviewing a Credit Report :: Organize entries identify possible errors and prepare official dispute documentation
Preparing Retirement Planning Questions :: Gather goals accounts assumptions fees and issues for a qualified professional
Organizing Tax Preparation Documents :: Build a complete records checklist and flag questions for a tax professional
Comparing Insurance Options :: Understand terms exclusions deductibles limits and questions without prescribing coverage
Responding to a Suspected Consumer Scam :: Preserve evidence secure accounts and identify official reporting channels
"""
        ),
        "focus": _focus(
            "financial goals|cash-flow tracking|assumption documentation|fees and total cost|scenario comparison|risk tolerance questions|records and evidence|consumer protections|privacy and fraud prevention|professional review points|decision checkpoints|ongoing monitoring"
        ),
    },
    "Marketing and Content Strategy": {
        "family": "Professional",
        "projects": _projects(
            """
Defining a Useful Audience Profile :: Ground messaging in real needs contexts objections and responsible evidence
Building a Multi-Channel Content Calendar :: Coordinate themes formats owners deadlines reuse and review
Designing an Email Campaign :: Plan segmentation value subject lines sequencing accessibility and measurement
Improving Search-Friendly Website Content :: Align user intent helpful structure technical basics and honest claims
Creating a Social Media Workflow :: Balance consistency engagement moderation platform differences and team capacity
Clarifying Product Messaging :: Explain the problem value differentiators evidence limitations and next step
Promoting a Community Event :: Coordinate audiences partners schedule channels accessibility and updates
Analyzing Customer Feedback Themes :: Code recurring needs compare segments and turn evidence into prioritized actions
Creating a Consistent Brand Voice Guide :: Define principles examples exceptions and review practices across writers
Reviewing Campaign Performance :: Interpret metrics attribution limits qualitative feedback and next experiments
"""
        ),
        "focus": _focus(
            "audience research|message hierarchy|channel selection|content planning|brand voice|accessibility|claims and evidence|review workflow|campaign sequencing|measurement design|feedback analysis|iteration priorities"
        ),
    },
    "Design and Creative Projects": {
        "family": "Professional",
        "projects": _projects(
            """
Creating a Brand Identity Brief :: Translate purpose audience personality constraints and applications into design direction
Redesigning a Product User Flow :: Map user goals friction states errors accessibility and measurable improvements
Planning a Flexible Room Layout :: Compare activities dimensions movement storage lighting and changing household needs
Designing a Clear Presentation :: Build visual hierarchy pacing evidence and audience-appropriate storytelling
Creating an Informational Infographic :: Turn accurate data into a readable ethical and accessible visual narrative
Planning a Documentary Photo Project :: Define story access consent shot needs schedule and editing approach
Storyboarding a Short Video :: Coordinate scenes narration visuals timing transitions and production constraints
Developing an Illustration Series :: Maintain a coherent concept style palette composition and delivery workflow
Selecting an Accessible Color System :: Balance identity contrast states themes and implementation requirements
Curating a Creative Portfolio :: Choose work explain process tailor sequencing and incorporate constructive critique
"""
        ),
        "focus": _focus(
            "creative brief|audience needs|concept exploration|visual hierarchy|composition and flow|color and typography|accessibility|asset organization|prototype and critique|revision history|production constraints|delivery specifications"
        ),
    },
    "Personal Productivity and Organization": {
        "family": "Personal and Daily Life",
        "projects": _projects(
            """
Creating a Realistic Weekly Plan :: Balance priorities appointments focused work household needs and recovery time
Organizing a Large Task Backlog :: Clarify outcomes remove stale work sequence dependencies and limit active commitments
Cleaning Up Digital Files :: Design a simple naming folder archive search and backup system
Turning Meetings into Action :: Capture decisions owners deadlines dependencies and unresolved questions consistently
Reducing Email Overload :: Create triage response follow-up reference and unsubscribe routines
Building a Sustainable Habit System :: Connect cues actions rewards tracking recovery and changing circumstances
Planning a Household Move :: Coordinate dates decisions packing services records responsibilities and contingencies
Keeping a Decision Journal :: Record context options assumptions choices outcomes and later learning
Coordinating a Shared Family Calendar :: Resolve schedule conflicts responsibilities reminders and privacy boundaries
Building a Personal Reference Library :: Store trusted notes templates instructions and review dates for recurring needs
"""
        ),
        "focus": _focus(
            "priority definition|time estimation|dependency mapping|task ownership|calendar coordination|information capture|naming and retrieval|automation opportunities|habit cues and review|boundary setting|backup plans|weekly reflection"
        ),
    },
    "Travel and Local Planning": {
        "family": "Personal and Daily Life",
        "projects": _projects(
            """
Planning a Weekend City Visit :: Build a paced itinerary around interests location hours transport and rest
Preparing for an International Trip :: Coordinate documents timing communication money health questions and local norms
Designing a Scenic Road Trip :: Balance routes stops driving limits lodging fuel weather and alternatives
Creating an Accessible Travel Plan :: Verify mobility sensory dietary communication and accommodation needs directly
Organizing a Business Trip :: Align meetings travel buffers workspace receipts and contingency communication
Planning a Multi-Generation Family Vacation :: Balance ages interests energy budgets privacy and shared decisions
Building a Budget-Conscious Itinerary :: Compare total costs free options booking terms and flexible priorities
Creating a Reliable Packing System :: Match activities climate duration laundry technology documents and backups
Exploring Local Culture Respectfully :: Research history etiquette language community businesses and visitor impact
Responding to Travel Disruptions :: Replan connections lodging commitments documents and communications as facts change
"""
        ),
        "focus": _focus(
            "traveler preferences|date and time constraints|route planning|transport connections|lodging criteria|opening-hour verification|accessibility needs|budget tracking|documents and reservations|local etiquette|weather contingencies|communication plans"
        ),
    },
    "Home DIY and Maintenance": {
        "family": "Personal and Daily Life",
        "projects": _projects(
            """
Building a Seasonal Home Maintenance Schedule :: Track inspections cleaning replacements service records and reminders
Planning a Simple Wall Shelf Project :: Measure select materials identify tools sequence work and use safe mounting guidance
Troubleshooting a Household Appliance :: Gather symptoms model information and safe observations before professional service
Conducting a Basic Home Energy Review :: Identify usage patterns drafts insulation behaviors and qualified upgrade questions
Designing a Small Garden Irrigation Plan :: Match plant zones climate timing water efficiency maintenance and restrictions
Decluttering a Multi-Use Room :: Define functions categories storage limits donation decisions and maintenance habits
Coordinating a Household Move-In :: Prioritize utilities safety checks cleaning repairs furniture and address updates
Planning a Small Renovation :: Clarify scope budget sequence contractors permits decisions and change control
Setting Up Practical Smart-Home Devices :: Balance convenience interoperability privacy reliability and manual fallbacks
Creating a Household Emergency Kit :: Plan for local risks household needs rotation communication and official guidance
"""
        ),
        "focus": _focus(
            "scope and measurements|tool and material lists|safety boundaries|manufacturer instructions|maintenance history|cost estimates|task sequencing|contractor questions|permit checks|energy and waste|documentation|inspection and follow-up"
        ),
    },
    "Cooking and Meal Planning": {
        "family": "Personal and Daily Life",
        "projects": _projects(
            """
Planning a Flexible Week of Meals :: Balance schedule preferences nutrition variety leftovers and shopping effort
Cooking Creatively from Pantry Staples :: Inventory ingredients combine compatible flavors and avoid unnecessary waste
Organizing a Dinner Party :: Coordinate menu timing equipment allergies seating serving and cleanup
Learning Foundational Baking :: Understand ratios mixing fermentation temperature timing and diagnostic cues
Adapting Meals for Dietary Needs :: Verify restrictions prevent cross-contact and design satisfying shared alternatives
Reducing Household Food Waste :: Improve planning storage labeling reuse portions and compost decisions
Building a Batch-Cooking Routine :: Select recipes coordinate prep storage reheating and schedule variety
Learning a New Cooking Technique :: Progress from explanation to supervised practice observation and correction
Adapting a Favorite Recipe :: Change servings ingredients equipment or timing while preserving the intended result
Managing a Grocery Budget :: Plan flexible meals compare unit costs use inventory and update for price changes
"""
        ),
        "focus": _focus(
            "dietary preferences|allergy and cross-contact checks|ingredient inventory|substitutions|portion scaling|prep sequencing|temperature and timing|equipment constraints|food storage safety|shopping organization|leftover reuse|taste feedback"
        ),
    },
    "Relationships Parenting and Caregiving": {
        "family": "Personal and Daily Life",
        "projects": _projects(
            """
Coordinating a Busy Family Schedule :: Balance school work care appointments activities and shared responsibilities
Preparing for a Difficult Conversation :: Clarify goals observations feelings boundaries questions and a respectful setting
Improving Co-Parenting Logistics :: Document schedules exchanges expenses updates and child-centered contingencies
Organizing Eldercare Responsibilities :: Coordinate appointments records household tasks communication and respite support
Creating a Meaningful Relationship Ritual :: Design a sustainable shared practice around both people's preferences
Reflecting on a Recurring Conflict :: Separate facts interpretations patterns needs and possible repair steps
Coordinating Wedding Party Responsibilities :: Track roles dates clothing travel costs communication and backup plans
Drafting a Roommate Agreement :: Discuss shared spaces costs guests chores quiet time communication and review
Supporting a Child's Study Routine :: Build predictable structure encouragement choices feedback and school coordination
Helping a Friend through a Stressful Period :: Offer practical presence respect boundaries and encourage qualified support
"""
        ),
        "focus": _focus(
            "shared goals|individual preferences|roles and responsibilities|schedule coordination|clear requests|active listening|boundaries and consent|conflict de-escalation|child or dependent needs|support resources|privacy|follow-up agreements"
        ),
    },
    "Hobbies Games and Entertainment": {
        "family": "Personal and Daily Life",
        "projects": _projects(
            """
Building a Personalized Reading Plan :: Explore genres themes formats pacing availability and reflection without spoilers
Designing a Tabletop Role-Playing Campaign :: Develop setting characters conflicts clues sessions safety tools and continuity
Improving at Chess :: Review games identify patterns practice tactics study strategy and track progress
Planning a Photography Learning Project :: Combine composition light camera practice editing critique and a final series
Creating a Music Practice Routine :: Balance technique repertoire listening improvisation rest and performance goals
Starting a Beginner Woodworking Project :: Choose a suitable build learn safe tool use measure assemble finish and review
Researching Family History :: Organize known facts sources uncertainties interviews records and respectful privacy choices
Learning a New Craft :: Select materials practice core techniques diagnose mistakes and complete a meaningful object
Designing a Small Video Game :: Define a playable loop controls levels art sound testing scope and release plan
Hosting a Themed Movie or Game Night :: Coordinate preferences access content comfort timing food and group activities
"""
        ),
        "focus": _focus(
            "interest and skill baseline|equipment and materials|practice progression|creative constraints|rules and systems|session planning|feedback and critique|community participation|budget limits|safety practices|project documentation|showcase or performance"
        ),
    },
    "Community Events and Civic Life": {
        "family": "Civic and Administrative",
        "projects": _projects(
            """
Planning a Community Science Fair :: Coordinate participants exhibits judging venue schedule budget and accessibility
Organizing a Volunteer Service Day :: Match community needs tasks people tools safety communication and follow-up
Preparing a Neighborhood Meeting :: Build an inclusive agenda share evidence manage discussion and record decisions
Running a Small Fundraising Campaign :: Clarify purpose methods costs responsibilities donor communication and reporting
Designing a Public Workshop :: Define audience outcomes activities facilitators materials access and evaluation
Starting a Local Interest Club :: Establish purpose membership rhythm roles communication norms and sustainable activities
Preparing a Public Comment :: Understand the proposal organize evidence state impacts and meet submission requirements
Coordinating an Emergency Preparedness Drill :: Align official guidance roles scenarios communication access and debriefing
Creating a Community Garden Program :: Plan plots participation water tools rules education conflict handling and seasons
Producing a Cultural Festival :: Coordinate community partners artists vendors schedule permissions access and respect
"""
        ),
        "focus": _focus(
            "community needs|stakeholder mapping|roles and volunteers|venue and permits|budget and fundraising|schedule and logistics|accessibility and language access|public communication|safety planning|conflict handling|feedback collection|transparent reporting"
        ),
    },
    "Legal and Administrative Navigation": {
        "family": "Civic and Administrative",
        "projects": _projects(
            """
Organizing Questions about a Residential Lease :: Identify clauses dates records concerns and issues for qualified review
Preparing Small-Claims Documentation :: Build a factual chronology preserve evidence calculate documented amounts and check official rules
Creating an Immigration Paperwork Checklist :: Track official forms supporting records translations deadlines and professional questions
Preparing for an Estate-Planning Consultation :: Inventory goals assets relationships documents and questions for an attorney
Researching a Local Business License Process :: Find authoritative requirements sequence forms fees contacts and renewal dates
Organizing an Invention Prior-Art Search :: Document the concept search terms related work differences and patent-professional questions
Drafting a Consumer Complaint Record :: Present dates transactions promises evidence requested resolution and escalation channels
Preparing a Public-Benefits Application :: Follow official eligibility instructions gather records track submissions and request help
Planning a Public-Records Request :: Define the records agency scope date range format and applicable official process
Documenting an Insurance Claim :: Preserve incident facts evidence communications policy questions deadlines and adjustments
"""
        ),
        "focus": _focus(
            "authoritative source checks|jurisdiction and scope|document inventory|fact chronology|deadlines and notices|forms and submission steps|fees and receipts|evidence preservation|privacy and redaction|status tracking|professional review points|appeal or escalation questions"
        ),
    },
    "Philosophy Ethics and Decision-Making": {
        "family": "Personal and Learning",
        "projects": _projects(
            """
Evaluating AI Use in Hiring :: Weigh efficiency fairness privacy accountability evidence and less automated alternatives
Thinking through Personal Data Sharing :: Compare convenience consent power retention downstream use and reversibility
Discussing Free Will and Responsibility :: Explore competing philosophical views and their practical implications
Deciding How to Allocate a Scarce Resource :: Examine values stakeholders procedures consequences and legitimate disagreement
Evaluating Technology Use in Schools :: Balance learning access surveillance distraction teacher judgment and student voice
Considering the Ethics of Persuasive Design :: Analyze autonomy transparency vulnerability business incentives and safeguards
Discussing Animal Welfare Choices :: Compare ethical frameworks evidence practical constraints and gradual actions respectfully
Reasoning about Environmental Responsibility :: Explore individual institutional and policy duties across time and place
Handling a Conflict between Loyalty and Honesty :: Clarify relationships commitments harms alternatives and repair obligations
Evaluating an Automated Decision System :: Examine data assumptions explainability recourse oversight and distributional effects
"""
        ),
        "focus": _focus(
            "stakeholder identification|competing values|ethical frameworks|facts versus assumptions|rights and duties|consequences and trade-offs|fair procedures|power and vulnerability|transparency|accountability and recourse|reversibility|respectful disagreement"
        ),
    },
    "Recommendations and Shopping": {
        "family": "Personal and Daily Life",
        "projects": _projects(
            """
Choosing a Laptop for Real Needs :: Match workload portability battery repairability support and budget to available options
Selecting Comfortable Everyday Shoes :: Compare fit intended activity materials support durability and return policies
Finding Books for a Reading Mood :: Translate themes pace style length format and content preferences into suggestions
Choosing Family-Friendly Entertainment :: Balance ages interests access runtime content sensitivities and shared enjoyment
Comparing Home Office Chairs :: Evaluate fit adjustability dimensions materials warranty trials and price
Selecting a Beginner Camera Setup :: Match learning goals subjects portability controls lenses used options and budget
Choosing a Practical Kitchen Appliance :: Compare actual cooking habits capacity cleanup storage reliability and support
Finding a Suitable Learning Platform :: Assess curriculum practice feedback accessibility pacing privacy and cost
Comparing Phone Plans :: Analyze coverage usage international needs device terms fees support and flexibility
Choosing a Thoughtful Gift :: Use the recipient's interests relationship context constraints and delivery timing respectfully
"""
        ),
        "focus": _focus(
            "user needs|preference elicitation|must-have criteria|budget range|total ownership cost|compatibility|accessibility|independent evidence|availability and timing|warranty and returns|privacy considerations|shortlist comparison"
        ),
    },
    "Sports Fitness and Recreation": {
        "family": "Personal and Daily Life",
        "projects": _projects(
            """
Training for a First Recreational 5K :: Build gradually around baseline schedule recovery preferences and professional guidance
Improving a Casual Cycling Routine :: Balance routes skills equipment weather maintenance safety and progression
Learning Swimming Technique :: Organize coached practice cues drills comfort goals and safe supervision
Planning a Hiking Progression :: Match terrain distance weather equipment navigation group needs and turnaround rules
Developing Youth Team Practice Sessions :: Combine age-appropriate skills play inclusion feedback and safeguarding
Reviewing Personal Sports Performance :: Use observations and simple metrics to choose one improvement focus at a time
Returning to Activity after a Break :: Re-establish a conservative baseline monitor response and seek appropriate guidance
Organizing a Community Recreation League :: Coordinate rules teams facilities schedules officials access and communication
Learning Strength-Training Fundamentals :: Practice supervised technique sensible progression recovery and record keeping
Planning Outdoor Recreation for Mixed Abilities :: Adapt pace roles equipment routes communication and alternatives inclusively
"""
        ),
        "focus": _focus(
            "baseline and goals|skill progression|practice structure|equipment fit|warm-up and recovery|safe technique|weather and environment|accessibility adaptations|coaching feedback|schedule consistency|performance notes|professional guidance points"
        ),
    },
    "Customer Service and Workplace Communication": {
        "family": "Professional",
        "projects": _projects(
            """
Resolving a Complex Customer Complaint :: Build a clear timeline acknowledge impact verify policy and coordinate a fair response
Creating a Customer Support Knowledge Base :: Turn recurring issues into accurate searchable maintainable help content
Handling an Escalated Support Conversation :: De-escalate gather facts set expectations involve owners and document follow-up
Improving Internal Project Updates :: Communicate progress decisions risks dependencies and requests concisely
Preparing a Cross-Team Handoff :: Transfer context artifacts responsibilities open questions and verification steps
Designing a Service Recovery Process :: Define triage authority remedies communication measurement and learning
Writing Helpful Product Troubleshooting Replies :: Guide safe diagnostic steps without blaming users or overstating certainty
Coordinating Communication during an Outage :: Maintain accurate status cadence audience channels ownership and corrections
Building an Inclusive Meeting Practice :: Improve agendas participation access decisions action tracking and asynchronous input
Reviewing Workplace Communication Norms :: Align channel use response expectations tone confidentiality and escalation
"""
        ),
        "focus": _focus(
            "fact gathering|customer or colleague goals|tone and empathy|policy verification|clear expectations|ownership and escalation|response templates|decision records|status cadence|accessibility and inclusion|feedback loops|knowledge maintenance"
        ),
    },
}


HIGH_STAKES_CATEGORIES = {
    "Cybersecurity and Privacy",
    "Health and Wellness",
    "Personal Finance and Consumer Decisions",
    "Legal and Administrative Navigation",
}


def build_topics() -> list[dict[str, Any]]:
    selected_projects = []
    for project_index in range(10):
        for category, spec in DOMAIN_SPECS.items():
            projects = spec["projects"]
            focus_areas = spec["focus"]
            if len(projects) != 10:
                raise ValueError(f"{category} must contain exactly ten base projects")
            if len(focus_areas) < 10:
                raise ValueError(f"{category} must contain at least ten focus areas")
            selected_projects.append(
                (category, spec, project_index, projects[project_index])
            )
            if len(selected_projects) == 250:
                break
        if len(selected_projects) == 250:
            break

    topics = []
    next_id = 1
    for base_topic_id, selected in enumerate(selected_projects, start=1):
        category, spec, project_index, project = selected
        focus_areas = spec["focus"]
        base_title, goal = project
        for variant_index, variant in enumerate(SCENARIO_VARIANTS):
            rotation = (project_index * 2 + variant_index) % len(focus_areas)
            rotated = focus_areas[rotation:] + focus_areas[:rotation]
            subtopics = list(variant["angles"]) + rotated[:6]
            safety_note = ""
            if category in HIGH_STAKES_CATEGORIES:
                safety_note = (
                    " Keep assistance educational and organizational, use authoritative "
                    "sources, and route personalized decisions to qualified professionals."
                )
            topics.append(
                {
                    "id": next_id,
                    "category": category,
                    "title": f"{base_title} — {variant['label']}",
                    "theme": f"{goal}. {variant['theme']}{safety_note}",
                    "subtopics": subtopics,
                    "source_topic_id": base_topic_id,
                    "variant": variant_index + 1,
                    "variant_label": variant["label"],
                    "domain_family": spec["family"],
                }
            )
            next_id += 1
    return topics


def validate_topics(topics: Any) -> dict[str, Any]:
    errors = []
    if not isinstance(topics, list):
        raise ValueError("Topics must be a JSON array")
    if len(topics) != EXPECTED_TOPICS:
        errors.append(f"expected {EXPECTED_TOPICS} topics; found {len(topics)}")

    ids = [item.get("id") for item in topics if isinstance(item, dict)]
    titles = [item.get("title") for item in topics if isinstance(item, dict)]
    if ids != list(range(1, len(topics) + 1)):
        errors.append("topic IDs must be sequential from 1")
    if len(set(titles)) != len(topics):
        errors.append("topic titles must be unique")

    required = ("category", "title", "theme", "subtopics")
    for index, topic in enumerate(topics, start=1):
        if not isinstance(topic, dict):
            errors.append(f"topic {index} is not an object")
            continue
        missing = [field for field in required if not topic.get(field)]
        if missing:
            errors.append(f"topic {index} is missing: {', '.join(missing)}")
        subtopics = topic.get("subtopics")
        if (
            not isinstance(subtopics, list)
            or len(subtopics) != 8
            or not all(isinstance(item, str) and item.strip() for item in subtopics)
            or len(set(subtopics)) != len(subtopics)
        ):
            errors.append(f"topic {index} must have eight distinct subtopics")

    category_counts = Counter(
        topic["category"] for topic in topics if isinstance(topic, dict)
    )
    if set(category_counts) != set(DOMAIN_SPECS):
        errors.append("category set does not match the domain catalog")
    if any(count not in {80, 90} for count in category_counts.values()):
        errors.append("each category must contain 80 or 90 topics")

    variant_counts = Counter(
        topic.get("variant_label") for topic in topics if isinstance(topic, dict)
    )
    expected_variants = {item["label"] for item in SCENARIO_VARIANTS}
    if set(variant_counts) != expected_variants:
        errors.append("variant set does not match the scenario catalog")
    if any(count != 250 for count in variant_counts.values()):
        errors.append("each scenario variant must occur exactly 250 times")

    if errors:
        raise ValueError("Topic validation failed:\n- " + "\n- ".join(errors[:30]))

    return {
        "topics": len(topics),
        "projected_probe_rows": len(topics) * PROBES_PER_CHAT,
        "categories": len(category_counts),
        "base_topics": len(topics) // len(SCENARIO_VARIANTS),
        "variants": len(variant_counts),
        "category_counts": dict(category_counts),
        "variant_counts": dict(variant_counts),
        "family_counts": dict(Counter(topic["domain_family"] for topic in topics)),
        "unique_titles": len(set(titles)),
    }


def _inspiration_summary() -> dict[str, Any]:
    items = []
    available_paths = []
    for path in INSPIRATION_PATHS:
        if path.exists():
            available_paths.append(str(path))
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                items.extend(item for item in data if isinstance(item, dict))
    return {
        "files": available_paths,
        "topics_reviewed": len(items),
        "unique_titles_reviewed": len(
            {item.get("title") for item in items if item.get("title")}
        ),
        "source_categories": sorted(
            {str(item.get("category")) for item in items if item.get("category")}
        ),
    }


def prepare_topics(
    output: Path = DEFAULT_OUTPUT, report_output: Path = DEFAULT_REPORT
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    topics = build_topics()
    report = validate_topics(topics)
    report.update(
        {
            "status": "topics_ready_for_review",
            "generation_started": False,
            "probes_per_chat": PROBES_PER_CHAT,
            "inspiration": _inspiration_summary(),
            "safety_note": (
                "Health, finance, legal, privacy, and security topics are framed for "
                "educational, organizational, and defensive assistance."
            ),
        }
    )
    for path, value in ((output, topics), (report_output, report)):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(value, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    return topics, report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report-output", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    _, generated_report = prepare_topics(args.output, args.report_output)
    print(json.dumps(generated_report, indent=2, ensure_ascii=False))
