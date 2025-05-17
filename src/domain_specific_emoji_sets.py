"""
Domain-Specific Emoji Sets for specialized communication contexts.

This module provides specialized emoji vocabularies and patterns for
professional and technical domains, allowing for more precise emoji
communication in specific contexts.
"""

from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field

@dataclass
class DomainEmojiMapping:
    """Mapping between domain concepts and emoji representations."""
    emoji: str
    concept: str
    description: str
    usage_examples: List[str] = field(default_factory=list)
    related_concepts: List[str] = field(default_factory=list)

@dataclass
class EmojiSequence:
    """A meaningful sequence of emojis in a domain-specific context."""
    sequence: str
    meaning: str
    context: str
    possible_responses: List[str] = field(default_factory=list)


class DomainSpecificEmojiSet:
    """Base class for domain-specific emoji vocabularies."""
    
    def __init__(self, domain_name: str):
        """Initialize a domain-specific emoji set."""
        self.domain_name = domain_name
        self.emoji_mappings: Dict[str, DomainEmojiMapping] = {}
        self.common_sequences: List[EmojiSequence] = []
        
    def add_emoji_mapping(self, mapping: DomainEmojiMapping) -> None:
        """Add an emoji mapping to the domain set."""
        self.emoji_mappings[mapping.emoji] = mapping
        
    def add_common_sequence(self, sequence: EmojiSequence) -> None:
        """Add a common emoji sequence to the domain set."""
        self.common_sequences.append(sequence)
        
    def get_emoji_for_concept(self, concept: str) -> Optional[str]:
        """Get an emoji that represents a specific concept in this domain."""
        for emoji, mapping in self.emoji_mappings.items():
            if mapping.concept.lower() == concept.lower() or concept.lower() in [c.lower() for c in mapping.related_concepts]:
                return emoji
        return None
    
    def interpret_sequence(self, sequence: str) -> Optional[str]:
        """Interpret the meaning of an emoji sequence in this domain context."""
        for seq in self.common_sequences:
            if seq.sequence == sequence:
                return seq.meaning
        return None
    
    def suggest_response(self, sequence: str) -> List[str]:
        """Suggest appropriate responses to an emoji sequence."""
        for seq in self.common_sequences:
            if seq.sequence == sequence:
                return seq.possible_responses
        return []
    
    def get_all_concepts(self) -> List[str]:
        """Get all domain concepts with emoji representations."""
        concepts = []
        for mapping in self.emoji_mappings.values():
            concepts.append(mapping.concept)
            concepts.extend(mapping.related_concepts)
        return list(set(concepts))


class TechnicalSupportEmojiSet(DomainSpecificEmojiSet):
    """Emoji set for technical support and IT-related communications."""
    
    def __init__(self):
        """Initialize the technical support emoji set."""
        super().__init__("Technical Support")
        self._initialize_mappings()
        self._initialize_sequences()
        
    def _initialize_mappings(self):
        """Initialize technical support emoji mappings."""
        # Error states
        self.add_emoji_mapping(DomainEmojiMapping(
            emoji="🛑",
            concept="Critical Error",
            description="Indicates a critical system error that needs immediate attention",
            usage_examples=["🛑 Database down", "Server 🛑 needs reboot"],
            related_concepts=["System Failure", "Blocking Issue", "Crash"]
        ))
        
        self.add_emoji_mapping(DomainEmojiMapping(
            emoji="⚠️",
            concept="Warning",
            description="Indicates a non-critical warning that should be addressed soon",
            usage_examples=["⚠️ Disk space 85%", "Memory usage ⚠️"],
            related_concepts=["Alert", "Caution", "Attention Required"]
        ))
        
        self.add_emoji_mapping(DomainEmojiMapping(
            emoji="❓",
            concept="Unknown Error",
            description="Indicates an error with unknown cause or unexpected behavior",
            usage_examples=["System restarting ❓", "Login failing ❓"],
            related_concepts=["Investigation Needed", "Troubleshooting"]
        ))
        
        # System components
        self.add_emoji_mapping(DomainEmojiMapping(
            emoji="💾",
            concept="Database",
            description="Represents database systems or storage-related components",
            usage_examples=["💾 Backup complete", "💾 corruption detected"],
            related_concepts=["Storage", "Data", "SQL"]
        ))
        
        self.add_emoji_mapping(DomainEmojiMapping(
            emoji="🖥️",
            concept="Server",
            description="Represents server hardware or server applications",
            usage_examples=["🖥️ restart required", "🖥️ CPU at 95%"],
            related_concepts=["Hardware", "Host", "VM"]
        ))
        
        self.add_emoji_mapping(DomainEmojiMapping(
            emoji="📱",
            concept="Mobile Device",
            description="Represents mobile devices or mobile applications",
            usage_examples=["📱 app crashing", "📱 update available"],
            related_concepts=["Smartphone", "Tablet", "App"]
        ))
        
        self.add_emoji_mapping(DomainEmojiMapping(
            emoji="🔌",
            concept="Network/Connectivity",
            description="Represents network infrastructure or connectivity issues",
            usage_examples=["🔌 down in building 3", "VPN 🔌 issues"],
            related_concepts=["Internet", "WiFi", "Connection", "VPN"]
        ))
        
        # Actions/Solutions
        self.add_emoji_mapping(DomainEmojiMapping(
            emoji="🔄",
            concept="Restart/Refresh",
            description="Indicates a restart or refresh operation",
            usage_examples=["🔄 server now", "System needs 🔄"],
            related_concepts=["Reboot", "Cycle", "Reset"]
        ))
        
        self.add_emoji_mapping(DomainEmojiMapping(
            emoji="🔍",
            concept="Investigation",
            description="Indicates investigation or detailed examination needed",
            usage_examples=["🔍 logs for errors", "Need to 🔍 further"],
            related_concepts=["Debug", "Root Cause Analysis", "Troubleshooting"]
        ))
        
        self.add_emoji_mapping(DomainEmojiMapping(
            emoji="✅",
            concept="Resolved",
            description="Indicates an issue has been successfully resolved",
            usage_examples=["Bug #1234 ✅", "Network issue ✅"],
            related_concepts=["Fixed", "Completed", "Closed"]
        ))
        
        self.add_emoji_mapping(DomainEmojiMapping(
            emoji="🔒",
            concept="Security",
            description="Represents security features or security issues",
            usage_examples=["🔒 breach detected", "Update 🔒 patches"],
            related_concepts=["Authentication", "Encryption", "Firewall"]
        ))
        
        self.add_emoji_mapping(DomainEmojiMapping(
            emoji="⏱️",
            concept="Performance",
            description="Represents performance metrics or performance issues",
            usage_examples=["Website ⏱️ slow", "Database ⏱️ optimized"],
            related_concepts=["Speed", "Latency", "Optimization"]
        ))
        
    def _initialize_sequences(self):
        """Initialize common technical support emoji sequences."""
        # Critical system error sequence
        self.add_common_sequence(EmojiSequence(
            sequence="🛑🖥️⚡",
            meaning="Critical server power issue",
            context="Used when a server has power-related critical failures",
            possible_responses=["🔍🔌", "🔄🖥️"]
        ))
        
        # Network troubleshooting sequence
        self.add_common_sequence(EmojiSequence(
            sequence="❓🔌🔍",
            meaning="Investigating unknown network issue",
            context="Used when there's an unidentified connectivity problem",
            possible_responses=["🔄🔌", "🔍📊"]
        ))
        
        # Database backup sequence
        self.add_common_sequence(EmojiSequence(
            sequence="💾⏱️✅",
            meaning="Database backup completed successfully",
            context="Used to report a successful time-sensitive database backup",
            possible_responses=["👍", "📊"]
        ))
        
        # Security issue sequence
        self.add_common_sequence(EmojiSequence(
            sequence="⚠️🔒🔍",
            meaning="Security warning under investigation",
            context="Used when investigating a potential security breach",
            possible_responses=["🛑🔒", "🔍🔒⏱️"]
        ))
        
        # Mobile app deployment sequence
        self.add_common_sequence(EmojiSequence(
            sequence="📱🔄✅",
            meaning="Mobile app update deployed successfully",
            context="Used when a mobile app update has been deployed",
            possible_responses=["📊📱", "👍"]
        ))


class ProjectManagementEmojiSet(DomainSpecificEmojiSet):
    """Emoji set for project management and team coordination."""
    
    def __init__(self):
        """Initialize the project management emoji set."""
        super().__init__("Project Management")
        self._initialize_mappings()
        self._initialize_sequences()
        
    def _initialize_mappings(self):
        """Initialize project management emoji mappings."""
        # Task status
        self.add_emoji_mapping(DomainEmojiMapping(
            emoji="🆕",
            concept="New Task",
            description="Indicates a newly created task or story",
            usage_examples=["🆕 User login feature", "Payment integration 🆕"],
            related_concepts=["Backlog Item", "Story", "Ticket"]
        ))
        
        self.add_emoji_mapping(DomainEmojiMapping(
            emoji="🏗️",
            concept="In Progress",
            description="Indicates a task currently being worked on",
            usage_examples=["🏗️ Database migration", "UI redesign 🏗️"],
            related_concepts=["Working", "Development", "Ongoing"]
        ))
        
        self.add_emoji_mapping(DomainEmojiMapping(
            emoji="✅",
            concept="Completed",
            description="Indicates a task has been completed",
            usage_examples=["✅ Login page", "API integration ✅"],
            related_concepts=["Done", "Finished", "Resolved"]
        ))
        
        self.add_emoji_mapping(DomainEmojiMapping(
            emoji="⏸️",
            concept="On Hold",
            description="Indicates a task that has been paused or put on hold",
            usage_examples=["⏸️ Analytics module", "Social login ⏸️"],
            related_concepts=["Blocked", "Paused", "Suspended"]
        ))
        
        # Priority levels
        self.add_emoji_mapping(DomainEmojiMapping(
            emoji="🔥",
            concept="High Priority",
            description="Indicates a task with high priority that needs immediate attention",
            usage_examples=["🔥 Fix payment bug", "Security issue 🔥"],
            related_concepts=["Urgent", "Critical", "P1"]
        ))
        
        self.add_emoji_mapping(DomainEmojiMapping(
            emoji="⬆️",
            concept="Medium Priority",
            description="Indicates a task with medium priority",
            usage_examples=["⬆️ Improve search", "Refactor code ⬆️"],
            related_concepts=["Important", "P2"]
        ))
        
        self.add_emoji_mapping(DomainEmojiMapping(
            emoji="⬇️",
            concept="Low Priority",
            description="Indicates a task with low priority",
            usage_examples=["⬇️ Update docs", "Minor UI fix ⬇️"],
            related_concepts=["Nice-to-have", "P3"]
        ))
        
        # Time and deadlines
        self.add_emoji_mapping(DomainEmojiMapping(
            emoji="⏰",
            concept="Deadline",
            description="Indicates a task with an approaching deadline",
            usage_examples=["⏰ Report due Friday", "Release ⏰ tomorrow"],
            related_concepts=["Due Date", "Timeline", "Target Date"]
        ))
        
        self.add_emoji_mapping(DomainEmojiMapping(
            emoji="⌛",
            concept="Time-consuming",
            description="Indicates a task that requires significant time to complete",
            usage_examples=["⌛ Database migration", "Testing suite ⌛"],
            related_concepts=["Complex", "Long-running", "Extensive"]
        ))
        
        # Project artifacts
        self.add_emoji_mapping(DomainEmojiMapping(
            emoji="📊",
            concept="Data/Metrics",
            description="Represents data, analytics, or metrics",
            usage_examples=["📊 User growth", "Conversion rates 📊"],
            related_concepts=["Analytics", "Statistics", "Charts"]
        ))
        
        self.add_emoji_mapping(DomainEmojiMapping(
            emoji="📝",
            concept="Documentation",
            description="Represents project documentation or notes",
            usage_examples=["📝 API docs updated", "Need to update 📝"],
            related_concepts=["Notes", "Specs", "Requirements"]
        ))
        
        self.add_emoji_mapping(DomainEmojiMapping(
            emoji="👥",
            concept="Team/Collaboration",
            description="Represents team activities or collaboration needs",
            usage_examples=["👥 Meeting at 3pm", "Need 👥 input on design"],
            related_concepts=["Group", "Stakeholders", "Participants"]
        ))
        
        self.add_emoji_mapping(DomainEmojiMapping(
            emoji="🔄",
            concept="Iteration",
            description="Represents iteration or a repeated process",
            usage_examples=["🔄 Sprint 5 starting", "Feature needs 🔄"],
            related_concepts=["Cycle", "Sprint", "Version"]
        ))
        
    def _initialize_sequences(self):
        """Initialize common project management emoji sequences."""
        # High priority task assignment
        self.add_common_sequence(EmojiSequence(
            sequence="🆕🔥👤",
            meaning="New high-priority task assigned",
            context="Used when assigning a critical new task to a team member",
            possible_responses=["👍⏰", "🏗️🔍"]
        ))
        
        # Sprint planning sequence
        self.add_common_sequence(EmojiSequence(
            sequence="🔄📝👥",
            meaning="Sprint planning meeting",
            context="Used to coordinate a sprint planning session with the team",
            possible_responses=["👍⏰", "📊🔍"]
        ))
        
        # Deadline approaching sequence
        self.add_common_sequence(EmojiSequence(
            sequence="⏰🔥⚠️",
            meaning="Urgent deadline approaching",
            context="Used to highlight an imminent critical deadline",
            possible_responses=["🏗️⏱️", "👥🆘"]
        ))
        
        # Completed milestone sequence
        self.add_common_sequence(EmojiSequence(
            sequence="✅📊🎉",
            meaning="Milestone completed with positive metrics",
            context="Used to celebrate a completed project milestone with good results",
            possible_responses=["👏👏", "🔄⬆️"]
        ))
        
        # Blocked task sequence
        self.add_common_sequence(EmojiSequence(
            sequence="⏸️❓🔍",
            meaning="Task blocked, investigating cause",
            context="Used when a task is on hold pending investigation",
            possible_responses=["👥🔍", "⏱️🔄"]
        ))


class EducationalEmojiSet(DomainSpecificEmojiSet):
    """Emoji set for educational contexts and learning activities."""
    
    def __init__(self):
        """Initialize the educational emoji set."""
        super().__init__("Educational")
        self._initialize_mappings()
        self._initialize_sequences()
        
    def _initialize_mappings(self):
        """Initialize educational emoji mappings."""
        # Knowledge areas
        self.add_emoji_mapping(DomainEmojiMapping(
            emoji="🧮",
            concept="Mathematics",
            description="Represents mathematics or numerical concepts",
            usage_examples=["🧮 practice problems", "Need help with 🧮"],
            related_concepts=["Algebra", "Calculus", "Statistics"]
        ))
        
        self.add_emoji_mapping(DomainEmojiMapping(
            emoji="🧪",
            concept="Science",
            description="Represents scientific topics or experiments",
            usage_examples=["🧪 lab today", "🧪 research project"],
            related_concepts=["Chemistry", "Biology", "Physics", "Experiments"]
        ))
        
        self.add_emoji_mapping(DomainEmojiMapping(
            emoji="📚",
            concept="Literature",
            description="Represents literary studies or reading assignments",
            usage_examples=["📚 essay due Friday", "Read 📚 chapter 5"],
            related_concepts=["Reading", "Books", "Writing", "Essays"]
        ))
        
        self.add_emoji_mapping(DomainEmojiMapping(
            emoji="💻",
            concept="Computer Science",
            description="Represents computer science or programming topics",
            usage_examples=["💻 coding assignment", "Need help with 💻 project"],
            related_concepts=["Programming", "Coding", "Development", "Software"]
        ))
        
        self.add_emoji_mapping(DomainEmojiMapping(
            emoji="🌎",
            concept="Geography/History",
            description="Represents geography, history, or social studies",
            usage_examples=["🌎 quiz tomorrow", "🌎 research paper"],
            related_concepts=["History", "Social Studies", "Maps", "Culture"]
        ))
        
        # Learning activities
        self.add_emoji_mapping(DomainEmojiMapping(
            emoji="📝",
            concept="Notes/Writing",
            description="Represents note-taking or writing activities",
            usage_examples=["📝 lecture notes", "📝 draft due Monday"],
            related_concepts=["Essays", "Notes", "Documentation", "Summaries"]
        ))
        
        self.add_emoji_mapping(DomainEmojiMapping(
            emoji="📖",
            concept="Reading",
            description="Represents reading assignments or materials",
            usage_examples=["📖 chapters 3-4", "Complete 📖 by Friday"],
            related_concepts=["Textbooks", "Articles", "Literature", "Study"]
        ))
        
        self.add_emoji_mapping(DomainEmojiMapping(
            emoji="🧠",
            concept="Comprehension/Understanding",
            description="Represents comprehension or deep understanding",
            usage_examples=["🧠 complex concepts", "Need to 🧠 this topic"],
            related_concepts=["Learning", "Mastery", "Cognition", "Thinking"]
        ))
        
        self.add_emoji_mapping(DomainEmojiMapping(
            emoji="🔬",
            concept="Research/Investigation",
            description="Represents research or investigative activities",
            usage_examples=["🔬 project proposal", "Working on 🔬 thesis"],
            related_concepts=["Analysis", "Experimentation", "Discovery", "Study"]
        ))
        
        # Feedback and assessment
        self.add_emoji_mapping(DomainEmojiMapping(
            emoji="✅",
            concept="Correct/Complete",
            description="Indicates correct answers or completed assignments",
            usage_examples=["Question 3 ✅", "Assignment ✅ submitted"],
            related_concepts=["Correct", "Accurate", "Finished", "Complete"]
        ))
        
        self.add_emoji_mapping(DomainEmojiMapping(
            emoji="❌",
            concept="Incorrect/Incomplete",
            description="Indicates incorrect answers or incomplete work",
            usage_examples=["Problem 2 ❌", "❌ missing references"],
            related_concepts=["Wrong", "Error", "Mistake", "Incomplete"]
        ))
        
        self.add_emoji_mapping(DomainEmojiMapping(
            emoji="📊",
            concept="Assessment/Grade",
            description="Represents assessments, grades, or evaluations",
            usage_examples=["📊 test results", "Project 📊 feedback"],
            related_concepts=["Grades", "Scores", "Evaluation", "Results"]
        ))
        
        self.add_emoji_mapping(DomainEmojiMapping(
            emoji="🔍",
            concept="Review/Revision",
            description="Represents review or revision activities",
            usage_examples=["🔍 essay draft", "Need to 🔍 before submission"],
            related_concepts=["Edit", "Proofread", "Revise", "Improve"]
        ))
        
        # Time management and organization
        self.add_emoji_mapping(DomainEmojiMapping(
            emoji="⏰",
            concept="Deadline",
            description="Indicates assignment deadlines or time constraints",
            usage_examples=["Essay ⏰ Friday", "Project ⏰ approaching"],
            related_concepts=["Due Date", "Time Limit", "Submission"]
        ))
        
        self.add_emoji_mapping(DomainEmojiMapping(
            emoji="📅",
            concept="Schedule",
            description="Represents schedules or calendar planning",
            usage_examples=["📅 study group", "Exam 📅 next week"],
            related_concepts=["Calendar", "Timetable", "Planning", "Dates"]
        ))
        
    def _initialize_sequences(self):
        """Initialize common educational emoji sequences."""
        # Assignment guidance sequence
        self.add_common_sequence(EmojiSequence(
            sequence="📝🔍✅",
            meaning="Writing assignment feedback with corrections",
            context="Used when providing feedback on written work",
            possible_responses=["👍🔍", "❓📝"]
        ))
        
        # Study session sequence
        self.add_common_sequence(EmojiSequence(
            sequence="📚🧠⏰",
            meaning="Intensive study session with time constraints",
            context="Used to organize focused study time before a deadline",
            possible_responses=["👍📅", "❓📚"]
        ))
        
        # Science project sequence
        self.add_common_sequence(EmojiSequence(
            sequence="🧪🔬📊",
            meaning="Science experiment with data analysis",
            context="Used to describe a scientific experiment requiring data analysis",
            possible_responses=["🧠❓", "📝✅"]
        ))
        
        # Group project sequence
        self.add_common_sequence(EmojiSequence(
            sequence="👥💻📅",
            meaning="Programming group project scheduling",
            context="Used to coordinate team coding project timelines",
            possible_responses=["👍⏰", "📝🔍"]
        ))
        
        # Exam preparation sequence
        self.add_common_sequence(EmojiSequence(
            sequence="📖🧮⚠️",
            meaning="Urgent math study needed",
            context="Used to highlight critical math review before evaluation",
            possible_responses=["👍🧠", "❓🧮"]
        ))


class FinancialEmojiSet(DomainSpecificEmojiSet):
    """Emoji set for financial discussions and transactions."""
    
    def __init__(self):
        """Initialize the financial emoji set."""
        super().__init__("Financial")
        self._initialize_mappings()
        self._initialize_sequences()
        
    def _initialize_mappings(self):
        """Initialize financial emoji mappings."""
        # Currencies and money
        self.add_emoji_mapping(DomainEmojiMapping(
            emoji="💰",
            concept="Money/Cash",
            description="Represents money or cash in general",
            usage_examples=["💰 received", "Need 💰 for project"],
            related_concepts=["Funds", "Capital", "Cash", "Finances"]
        ))
        
        self.add_emoji_mapping(DomainEmojiMapping(
            emoji="💵",
            concept="Dollar",
            description="Represents US dollars or dollar-denominated transactions",
            usage_examples=["💵 500 payment", "Convert to 💵"],
            related_concepts=["USD", "US Currency", "American Dollar"]
        ))
        
        self.add_emoji_mapping(DomainEmojiMapping(
            emoji="💶",
            concept="Euro",
            description="Represents euros or euro-denominated transactions",
            usage_examples=["💶 200 transfer", "Price in 💶"],
            related_concepts=["EUR", "European Currency"]
        ))
        
        self.add_emoji_mapping(DomainEmojiMapping(
            emoji="💷",
            concept="Pound",
            description="Represents British pounds or pound-denominated transactions",
            usage_examples=["💷 invoice paid", "Convert to 💷"],
            related_concepts=["GBP", "British Currency", "Sterling"]
        ))
        
        self.add_emoji_mapping(DomainEmojiMapping(
            emoji="💴",
            concept="Yen",
            description="Represents Japanese yen or yen-denominated transactions",
            usage_examples=["💴 market rising", "Prices in 💴"],
            related_concepts=["JPY", "Japanese Currency"]
        ))
        
        # Transaction types
        self.add_emoji_mapping(DomainEmojiMapping(
            emoji="↗️",
            concept="Income/Inflow",
            description="Represents income, revenue, or money received",
            usage_examples=["↗️ monthly report", "Sales ↗️ 15%"],
            related_concepts=["Revenue", "Earnings", "Profit", "Gains"]
        ))
        
        self.add_emoji_mapping(DomainEmojiMapping(
            emoji="↘️",
            concept="Expense/Outflow",
            description="Represents expenses, costs, or money spent",
            usage_examples=["Marketing ↘️ $5000", "Monthly ↘️ report"],
            related_concepts=["Costs", "Spending", "Expenditure", "Outgoing"]
        ))
        
        self.add_emoji_mapping(DomainEmojiMapping(
            emoji="🔄",
            concept="Transfer",
            description="Represents transfers between accounts or entities",
            usage_examples=["🔄 to savings", "Wire 🔄 complete"],
            related_concepts=["Wire", "Move Funds", "Account Transfer"]
        ))
        
        self.add_emoji_mapping(DomainEmojiMapping(
            emoji="💳",
            concept="Credit/Card Payment",
            description="Represents credit card or card-based transactions",
            usage_examples=["💳 processing", "Pay with 💳"],
            related_concepts=["Card", "Credit Card", "Debit Card", "Electronic Payment"]
        ))
        
        # Financial concepts
        self.add_emoji_mapping(DomainEmojiMapping(
            emoji="📈",
            concept="Growth/Increase",
            description="Represents financial growth, increases, or upward trends",
            usage_examples=["Stock 📈 today", "Revenue 📈 this quarter"],
            related_concepts=["Profit", "Appreciation", "Gain", "Rise"]
        ))
        
        self.add_emoji_mapping(DomainEmojiMapping(
            emoji="📉",
            concept="Decline/Decrease",
            description="Represents financial decline, decreases, or downward trends",
            usage_examples=["Market 📉 2%", "Expenses 📉 after cuts"],
            related_concepts=["Loss", "Depreciation", "Drop", "Fall"]
        ))
        
        self.add_emoji_mapping(DomainEmojiMapping(
            emoji="⚖️",
            concept="Balance/Budget",
            description="Represents balanced finances or budgeting",
            usage_examples=["Accounts ⚖️", "Monthly ⚖️ review"],
            related_concepts=["Budget", "Reconciliation", "Break-even", "Accounting"]
        ))
        
        self.add_emoji_mapping(DomainEmojiMapping(
            emoji="📊",
            concept="Financial Report/Analytics",
            description="Represents financial reports, statements, or analytics",
            usage_examples=["Quarterly 📊", "Revenue 📊 analysis"],
            related_concepts=["Statement", "Report", "Analysis", "Metrics"]
        ))
        
        self.add_emoji_mapping(DomainEmojiMapping(
            emoji="💼",
            concept="Business/Investment",
            description="Represents business transactions or investments",
            usage_examples=["💼 opportunity", "New 💼 venture"],
            related_concepts=["Company", "Investment", "Venture", "Enterprise"]
        ))
        
        self.add_emoji_mapping(DomainEmojiMapping(
            emoji="🏦",
            concept="Bank/Financial Institution",
            description="Represents banks or financial institutions",
            usage_examples=["🏦 appointment", "Contact 🏦 support"],
            related_concepts=["Banking", "Credit Union", "Financial Services"]
        ))
        
        self.add_emoji_mapping(DomainEmojiMapping(
            emoji="📝",
            concept="Contract/Agreement",
            description="Represents financial contracts or agreements",
            usage_examples=["Loan 📝 signed", "Review 📝 terms"],
            related_concepts=["Terms", "Document", "Agreement", "Policy"]
        ))
        
        # Time-related financial concepts
        self.add_emoji_mapping(DomainEmojiMapping(
            emoji="⏰",
            concept="Deadline/Due Date",
            description="Represents payment deadlines or due dates",
            usage_examples=["Invoice ⏰ Friday", "⏰ for tax filing"],
            related_concepts=["Due Date", "Payment Deadline", "Timeframe"]
        ))
        
        self.add_emoji_mapping(DomainEmojiMapping(
            emoji="🔄",
            concept="Recurring/Subscription",
            description="Represents recurring payments or subscriptions",
            usage_examples=["Monthly 🔄 payment", "Cancel 🔄 service"],
            related_concepts=["Automatic Payment", "Subscription", "Recurring"]
        ))
        
    def _initialize_sequences(self):
        """Initialize common financial emoji sequences."""
        # Payment confirmation sequence
        self.add_common
