## What I Built
I built prompt-lab.html, a single-page app in my week-02 folder that calls the Anthropic API using claude-sonnet-5. I paste my API key into the app each session, so it never appears in any file. The Export button downloads each conversation as Markdown, with the model name on the first line and each message under a System, You, or Claude heading. I ran each of the four Wildframe tasks in its own conversation, using the same system prompt each time to set the company, customers, and brand voice. Everything specific to a technique, including roles, examples, data, and format requirements, went in the user message. That way the shared system prompt never added Brand Kit examples to any task.

## Why This Approach

Task 1 - Technique: zero-shot
The Brand Kit has no examples of subscriber perks, so a one-shot or few-shot prompt would have meant borrowing a bean card or customer reply that shows nothing about what a perk looks like. The required format (six numbered ideas, one sentence each, no more than 25 words) is simple enough to state directly, and examples tend to pull brainstormed ideas toward a single theme. The system prompt's voice description and a retention-manager role gave the model enough direction without anchoring its ideas.

Task 2 - Technique: one-shot
The hard part of this task is reproducing the house bean card exactly: the same labels, in the same order, with an all-caps name line. That format is easier to show than to describe, and the Brand Kit contains exactly one bean card, so one example was both all I had and all I needed. I also told the model not to reuse the example's Real talk joke, since one-shot prompts often copy the example's content along with its format.

Task 3 - Technique: few-shot
Brand voice is a mix of tone, rhythm, and humor that is hard to pin down with adjectives but easy to pick up from examples. The four past replies cover different situations (a skeptical question, a complaint, praise, and a how-to), and they map onto the three new comments, so the model could see how the voice shifts between apologizing and saying thanks. A single example would have taught only one of those registers.

Task 4 - Technique: chain-of-thought
Choosing a location takes multi-step arithmetic (buyers, revenue, and rent for each site, then four-week totals) plus a hard budget constraint, and models are prone to mistakes when they jump straight to an answer on problems like this. Asking for step-by-step reasoning, with all the location data included in the prompt, made every calculation visible and checkable. It also caught the trap in the data: Union Station has the most visitors, but its $8,000 rent breaks the $5,000 budget and it loses money every week.

## Example Run
Model: claude-sonnet-5

### System

You work for Wildframe Coffee Roasters, a specialty coffee roaster that sells beans online and through a monthly subscription. Our customers are environmentally conscious urban professionals. Our brand voice is warm, plainspoken, and a little irreverent, with no corporate jargon.

### You

As our community manager, reply to three new customer comments. Here are comments we've received before and the replies we posted:

Comment: Is your packaging actually recyclable, or is that just marketing?
Reply: Fair question. The bag is compostable, the little valve isn't, so pop the valve off first. We're working on that last bit.

Comment: Your Kenya roast tasted burnt to me.
Reply: That's not the cup we were going for. Tell us how you brewed it and we'll help you dial it in, or swap you to something mellower.

Comment: Best coffee I've had in years!!
Reply: This made our whole roasting crew's morning. We'll keep doing the thing.

Comment: Can I pause my subscription while I travel?
Reply: Yep. Account, then Subscription, then Pause. Your beans will wait for you, mostly patiently.

New comments:
1. My order was supposed to arrive Monday. It's Thursday.
2. Do you make any decaf that doesn't taste like cardboard?
3. The Huila is my new Sunday morning ritual. Thank you.

Format requirements:
- Three replies, numbered 1 to 3, matching the comments above
- Each reply is no more than 40 words
- No emoji
- Don't invent specific product names, policies, or promises (like refunds or delivery dates) that aren't given here
- No intro or closing text, just the three replies
- Match the voice of the past replies, but don't reuse their specific phrases or jokes
- For complaints, lead with ownership and a clear next step; keep humor for lighter comments

### Claude

1. That's on us, not the mail gods. Send us your order number and we'll find out exactly where it's stuck and get it moving.
2. We do, and we actually drink it ourselves, so cardboard was never on the table. Worth a try.
3. Huila on Sundays sounds like a solid life choice. Thanks for letting us be part of the ritual.

## Reflection
The technique choice made the biggest difference in Task 3. Few-shot captured Wildframe's voice better than any description could have, but it also brought in the examples' quirks. My first run replied to the late-order complaint by reusing "mostly patiently" from the subscription-pause example, which read as an inside joke at a frustrated customer's expense. I had to add constraints against reusing phrases and against humor in complaints, so few-shot gave me the voice but needed guardrails. Even the final run's decaf reply says "We do," which claims a product exists when nothing in the Brand Kit confirms it, so a real team would need to check that before posting. Task 2 showed the clearest one-shot payoff: a single example produced three correctly formatted cards, with the labels in order. Task 4's chain-of-thought response got every calculation right, and its reasoning went beyond the arithmetic, connecting whole-bean buyers at the farmers market to future subscribers. I'm least sure about Task 1. The zero-shot ideas met the format and fit the brand, but a few were predictable, like a discount and tree planting. Next time I'd try chain-of-thought there: first have the model reason about what a six-month subscriber values and what a small roaster can actually afford, then generate ideas from that analysis.