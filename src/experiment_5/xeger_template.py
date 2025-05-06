from typing import ClassVar
from autointent.generation.chat_templates import BaseSynthesizerTemplate
from autointent.generation.chat_templates import Message, Role

class ReSynthesizerTemplateEn(BaseSynthesizerTemplate):

    _INTENT_NAME_LABEL = "Intent Name"
    _EXAMPLE_UTTERANCES_LABEL = "Example Utterances"
    _GENERATE_INSTRUCTION = "Please generate {n_examples} regular expressions for this intent.\n"

    _MESSAGES_TEMPLATE: ClassVar[list[Message]] = [
        Message(
            role=Role.USER,
            content=(
                "You will be given a set of example utterances and the name of a general theme (intent). "
                "Your task is to generate regular expressions that match this intent.\n\n"
                "Rules:\n"
                "- A regular expression should be a sequence of parentheses separated by spaces\n"
                "- Each parenthesis contains alternative words separated by the '|' symbol that can fit in that position\n"
                "- If the intent name is missing, infer it from the examples\n"
                "- If the examples are missing, use only the intent name\n"
                "{extra_instructions}\n\n"
                "Intent Name: add the music\n\n"
                "Example Utterances:\n"
                "1. add this artist to my Sinfonía Hipster\n"
                "2. I want to put Land of the Dead into my Big Daddy's Booze & Blues playlist\n"
                "3. Put a track by lil mama into my guest list sneaky zebra playlist.\n"
                "4. add the tune by misato watanabe to the Trapeo playlist\n"
                "5. add this album to the playlist called dishwashing\n\n"
                "Please generate a regular expression for this intent."
            ),
        ),
        Message(
            role=Role.ASSISTANT,
            content=(
                "1. (I|) (want to|need to) (add|put) (this|those|that) (artist|Land of the Dead|a track|music|album) (to|into|on|in|) (my||this|that) (playlist|list|Sinfonía Hipster) (please|)\n"
            ),
        ),
        Message(
            role=Role.USER,
            content=(
                "Intent Name: booking\n\n"
                "Example Utterances:\n"
                "1. Book a restaurant with a spa in Connecticut\n"
                "2. I would like a restaurant reservation for this year for 4 people.\n"
                "3. I need a table in one hour from now at somewhere not far from LA\n"
                "4. book The Fry Bread House for seven in Olive\n"
                "5. Book the Gus Stevens Seafood Restaurant & Buccaneer Lounge in Papua New Guinea for one person.\n\n"
                "Please generate a regular expression for this intent."
            ),
        ),
        Message(
            role=Role.ASSISTANT,
            content=("1. (Please|) (at two am|) (I|) (need|would like|want|'d like|) (a reservation|book|eat|reserve) (me|) (a|the|) (restaurant|hotel|spot|breakfast brasserie) (for ten|for this year|at 9 pm|at two am|at six am|) (for 4 people|for four people|for a party of seven|)"),
        ),
        Message(
            role=Role.USER,
            content=(
                "Intent Name: Play a music\n\n"
                "Example Utterances:\n"
                "1. I want to hear some psychedelic rock\n"
                "2. I want to hear something from Post-punk Revival\n"
                "3. Play some music from 1995\n"
                "4. I wish to listen to some instrumental music.\n"
                "5. play The New Timer by Augustus Pablo with Itunes\n\n"
                "Please generate a regular expression for this intent."
            ),
        ),
        Message(
            role=Role.ASSISTANT,
            content=("1. (Please|) (I want|I need|I wish|) (to|) (play|hear|listen to) (a|the|) (Play some music from 1995|wendy james|from 2000 on spotify|music on Lastfm)"),
        ),
        Message(
            role=Role.USER,
            content=(
                "Intent Name: weather\n\n"
                "Example Utterances:\n"
                "1. Tell me the weather forecast for Douglas State Forest\n"
                "2. What will the weather be in Bernie ID?\n"
                "3. How hot will it be in Wisconsin on August fourth?\n"
                "4. Is it freezing on June 21st in Apshawa, South Africa?\n"
                "5. Will the weather in Falcon Heights call for rain?\n\n"
                "Please generate a regular expression for this intent."
            ),
        ),
        Message(
            role=Role.ASSISTANT,
            content=(
                "1. (Tell me|Say|) (What|Is it) (will|) (a|the) (freezing|weather|weather forecast) (for|be|in) (Douglas State Forest|Bernie ID|in Wisconsin|) (on August fourth|on Dec. 26, 2027|twenty-fourth, 2028|) (/?|)\n"
            ),
        ),
    ]