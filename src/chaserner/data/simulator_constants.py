SEED = 42

#TRAIN_DEV_TEST_SPLIT = (0.45, 0.3, 0.25,)
TRAIN_DEV_TEST_SPLIT = (0.25, 0.1, 0.1,)

TRAIN_TEMPLATES = [
    "please {task} by {date} {person}",
    "{person} to {task} by {date}",
    "{person} to {task}",
    "can you {task}?",
    "{person} to {task} before {date}",
    "{person} : {task}",
    "{person}, {task} please, by {date}",
    "{person} will {task}, the due date is {date}",
    "{person} can you {task} before {date}"]
DEV_TEMPLATES = [
    "would it be straightforward to {task}?",
    "{person} to {task} by {date}",
    "{person} to {task}",
    "{person} will {task}",
    "Take note, {person}. The deadline for {task} is {date}.",
    "{person}, can you {task}?",
    "{person} will {task} before {date}",
    "{person}, just a heads up! We're hoping the task to {task} will be done by {date}."
    "{person}, can you {task} and have that finalized by {date}?"]
TEST_TEMPLATES = [
    "please {task}",
    "{task} - {person}",
    "{person} to {task} by {date}",
    "{person} to {task}",
    "{person} will {task}",
    "We've set a target date of {date} for you to {task}, {person}.",
    "{person}, by {date} hopefully we can {task}",
]


LONG_TEMPLATES = [
    "hello {person}, I just wanted to check if you were planning to complete {task} by {date}.",
    "before {date} we need to finish the {task}, which is something that {person} is working on.",
    "hey {person}, would you please make sure you do the {task} no later than {date}.",
    "this {task} needs to get done by {date}. {person} please make sure this is completed.",
    "lets {task} done before {date}. {person}, would you mind owning that?",
    "{person}, the deadline for the {task} is approaching on {date}. Please ensure it's completed.",
    "We're counting on you, {person}, to wrap up {task} by {date}.",
    "I've assigned the {task} to {person}. It should be ready by {date}.",
    "Remember, {person}, we have {task} due on {date}.",
    "It's essential for the {task} to be completed by {date}. Can we count on you, {person}?",
    "We're aiming to finish the {task} by {date}. {person}, please provide regular updates.",
    "{person}, the {task} is a top priority and has a deadline of {date}. Please adhere to it.",
    "For our project timeline, {task} should be wrapped up by {date}. {person}, we're depending on your contributions.",
    "We need the {task} sorted out by {date}. {person}, can you ensure a smooth execution?",
    "Our deadline for {task} is fast approaching on {date}. {person}, please push to get it done.",
    "We're setting a firm cut-off on {date} for the {task}. {person}, your diligence is appreciated.",
    "{person}, the {task} is expected to be in the final stages by {date}. Please ensure we meet that target.",
    "The {date} deadline for {task} is non-negotiable. {person}, we're relying heavily on your efforts.",
    "{person}, we're anticipating the {task} completion by {date}. Your focus on this is crucial.",
    "For the {task}, the target deadline is {date}. Can we get a confirmation on this, {person}?",
    "{person}, given the importance of the {task}, we must stick to the {date} deadline.",
    "We're in the final stretch for {task} with a due date of {date}. {person}, your timely completion is essential.",
    "All efforts should be directed towards completing the {task} by {date}. {person}, your updates are vital.",
    "{person}, with the {date} deadline for {task}, we're hoping for a seamless execution.",
    "The team is looking at {date} for finalizing the {task}. {person}, your expertise is invaluable here.",
    "{person}, we've set our sights on {date} for the {task}. Let's achieve this milestone together.",
    "For the {task}, {person}, your commitment to the {date} deadline is deeply appreciated.",
    "{person}, let's ensure the {task} is in the green by {date}. We're counting on your skills.",
    "We've chalked out {date} as the D-day for {task}. {person}, all the best!",
    "{person}, a gentle reminder about the {task}. We're expecting it to be ready by {date}.",
    "The finish line for {task} is {date}. {person}, we're optimistic about your contributions.",
    "{person}, the {task} has a strict timeline ending on {date}. Please keep that in mind.",
    "Our next milestone is the {task} on {date}. {person}, we're eager to see it through.",
    "In line with our goals, {task} is set for {date}. {person}, we're confident in your abilities.",
    "The set trajectory for {task} points to a {date} deadline. {person}, let's stick to the plan.",
    "{person}, the {task} is of paramount importance with a due date on {date}. Please prioritize.",
    "As the key person for {task}, {person}, we're looking at a {date} completion.",
    "Our project hinges on the {task} due by {date}. {person}, your role is pivotal.",
    "Could you confirm, {person}, that {task} will be ready by {date}?",
    "I trust you'll have the {task} sorted out, {person}, by the given deadline on {date}.",
    "{person}, it's crucial we hit the {date} deadline for {task}.",
    "Looking forward to your update on the {task}, {person}. Remember, it's due on {date}.",
    "By {date}, the {task} should be completed. Can you handle that, {person}?",
    "The deadline for {task} is set for {date}. {person}, you're on it, right?",
    "We've set a target date of {date} for {task}. It's in your hands, {person}.",
    "The {task} is on the calendar for {date}. Awaiting your update, {person}.",
    "Targeting {date} for the completion of {task}. {person}, please keep us posted.",
    "The {task} is due on {date}. We're looking to you for this, {person}.",
    "Mark the date, {person}. {task} needs to be wrapped up by {date}.",
    "Our milestone for {task} is {date}. {person}, let's make sure we achieve it.",
    "Ensuring the {task} is done by {date} is critical. Are you on track, {person}?",
    "The {task} is set to be finalized on {date}. {person}, please prioritize this.",
    "{task} is on the docket for {date}. We're trusting you with this, {person}.",
    "By {date}, we're expecting the {task} to be off our checklist. {person}, you've got this!",
    "The {task} is on our radar for {date}. We're relying on your expertise, {person}.",
    "{person}, as you're aware, our {task} deadline is {date}. Please ensure everything's on schedule.",
    "{task} is a key deliverable for {date}. {person}, keep us in the loop with your progress.",
    "We've circled {date} on our calendar for {task}. All eyes on you, {person}."
]






TRAIN_TASK_CATALOG = ["complete the reports on the model latency",
                "look through the revenue mockup sheet",
                "write the review",
                "read the document",
                "test out the chaser upgrade",
                "finalize the onboarding",
                "do the report",
                "figure out the bug",
                "find the highest traffic requests",
                "fix the broken supply chain",
                "reformat the data for better results",
                "reach out to several reviewers and get feedback",
                "do the presentation on the project plan",
                "create a cr for the newest feature processing code",
                "write the document on business opportunity in each locale",
                "check the design spec for the security review",
                "determine how many issues are still pending based on current fixes already implemented",
                "gather feedback report from the beta testers",
                "finish proposal for the marketing campaign",
                "start a webpage template for our investors",
                "complete analysis of user engagement metrics",
                "create the roadmap for the Q4 product releases",
                "finish the final proposal",
                "add the code review of the API integrations"]

DEV_TASK_CATALOG = [
                "update the dashboard design for user analytics",
                "summarize survey results on customer satisfaction",
                "finish the video tutorial for the new users",
                "create a database schema for the inventory system",
                "do the evaluation of our competitors' products",
                "evaluate our competitors products"
                "debug log for the app crashes last week",
                "maintain the daily webinars",
                "debug slack channel issues",
                "investigate the dropped payment processing",
                "research a refactoring plan for the legacy codebase",
                "add in some test cases for the login module",
                "submit the whitepaper on our machine learning approach",
                "figure out the deployment strategy for the cloud infrastructure"]
TEST_TASK_CATALOG = [
                "verify correct pay rates",
                "investigate taskers that have flagged this issue",
                "read the report",
                "iterate on the presentation",
                "complete the budget breakdown",
                "finalize prototype of the mobile application",
                "summarize the team's weekly standups",
                "write the training manual for the interns",
                "do the illustrations for the user guide",
                "write 1 pager",
                "follow up on comms for incorrect rates",
                "start the optimization plan for the database queries",
                "take down meeting minutes from the last board meeting",
                "share the draft for the press release",
                "work on wireframes for the user profile page",
                "audit report for the last fiscal year",
                "make a blueprint of the proposed office expansion",
                "add some benchmark tests for the latest software version",
                "make an onboarding checklist for new hires",
                "finish the translation of the product documentation into Spanish",
                "make some kind of outreach plan for potential partners"
]

TRAIN_PERSON_CATALOG = [
                "dustin axman",
                "denise liu",
                "josh martow",
                "diego",
                "abraham",
                "elias adum",
                "mary sue",
                "johnathan doe",
                "robert ross",
                "eric lee",
                "dustin",
                "todd",
                "rebecca dogmire",
                "michael axman",
                "emma rodriguez",
                "noah wilson",
                "harper anderson",
                "xiao gong",
                "ran xue",
                "david",
                "jing",
                "benjamin thomas"]

DEV_PERSON_CATALOG = [
                "jackson jackson",
                "lily taylor",
                "aiden moore",
                "matthew white",
                "ella harris",
                "eric",
                "samuel martin",
                "gary",
                "madison thompson",
                "joseph hernandez",
                "ying",
                "grace hall",
                "william young",
                "olivia turner",
                "james hill",
                "fabian",
                "emily perez",
                "alexander flores",
                "anthony",
                "abigail nelson",
                "amelia smith",
                "oliver jones",
                "isabella johnson",
                "sophia williams",
                "mia brown",
                "charlotte davis",
                "liam garcia",
                "lucas miller",
                "ava martinez",
                "ethan clark",
                "zoe lewis",
                "max",
                "nathanial wright",
                "chelsea baker",
                "geoffrey allen",
                "claire mitchell",
                "victor gonzalez"
]
TEST_PERSON_CATALOG = [
                "katherine scott",
                "timothy evans",
                "ruby king",
                "albert ramirez",
                "gwendolyn carter",
                "lawrence roberts",
                "heather robinson",
                "franklin phillips",
                "angelica torres",
                "marvin peterson",
                "louise green",
                "raymond reed",
                "julia cruz",
                "rachel",
                "rebecca",
                "perry",
                "rohit",
                "donald bailey",
                "diane russell",
                "clarence coleman",
                "fiona morris",
                "howard hughes",
                "vishal",
                "prem",
                "sreekar",
                "cassandra patterson",
                "reginald gray",
                "penny simmons",
                "terrence woods",
                "belinda foster",
                "duane hawkins",
                "melody hayes"
]

#Add all of the first names only
#PERSON_CATALOG = PERSON_CATALOG + [person.split()[0] for person in PERSON_CATALOG]


TRAIN_DATE_CATALOG = ["october 20th 2023",
                "friday",
                "q2",
                "the end of this sprint",
                "november 4th",
                "the 20th",
                "7/20/2025",
                "8/1/1996",
                "3/1",
                "09-1995",
                "the end of the fiscal year",
                "july 7th, 2023",
                "12-25-2025",
                "march 31st",
                "the following thursday",
                "first day of the next month",
                "end of this month",
                "6/30/2022",
                "q3 2024",
                "next quarter",
                "start of next year",
                "mid-august",
                "next wednesday",
                "the following friday",
                "april 15th, 2024",
                "7/20",
                "6/15",
                "10/31/2023",
                "4-23",
                "thursday",
                "monday next week",
                "the last day of this month",
                "friday, EOD",
                "12/31"]

DEV_DATE_CATALOG = ["may 10th",
                "january of next year",
                "winter break",
                "5/30/2024",
                "01-01-2000",
                "second week of feb",
                "fall 2022",
                "the start of q4",
                "12/31/2003",
                "tuesday of the third week",
                "june 5th, 2015",
                "summer 2025",
                "02-2021",
                "04/2002",
                "the end of q2 2025",
                "wednesday of next week",
                "the end of the calendar year",
                "11/11",
                "7/4/2022",
                "6/30/2022",
                "q3 2024",
                "next quarter",
                "start of next year",
                "second quarter of 2023",
                "8/24",
                "9/15/2021",
                "10/15",
                "march 1st, 2023",
                "the first week of december",
                "end of the week",
                "next monday",
                "start of march 2024",
                "end of june 2023",
                "next friday",
                "second week of january",
                "4/10",
                "last friday of the month"]
TEST_DATE_CATALOG = ["9/30/2023",
                "start of next month",
                "end of the next week",
                "8/11/2024",
                "2/12/2014",
                "5/1",
                "second tuesday of october",
                "5/15/2023",
                "early september 2024",
                "late july 2022",
                "march 31st",
                "the following thursday",
                "first day of the next month",
                "end of this month",
                "q4",
                "9 AM",
                "the first day of q1",
                "the start of the quarter",
                "2/03/2023",
                "5 PM sharp",
                "close of business",
                "around 3:30 PM",
                "first thing in the morning",
                "end of day",
                "4-21",
                "10:15 AM",
                "2 PM on the dot",
                "11:45 AM, right before lunch",
                "4:30 PM",
                "mid-morning",
                "just after lunch, about 1:15 PM",
                "the start of business tomorrow",
                "6 PM",
                "11 AM",
                "4:45 PM",
                "sometime between 2 and 4 PM"
]