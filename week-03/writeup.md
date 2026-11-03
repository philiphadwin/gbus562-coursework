## What I Built
I built ask.py, a Python command-line RAG tool that answers questions from a folder of documents. It splits each document into one chunk per section, stores the chunks in a local Chroma database using Chroma's built-in embedding model, retrieves the most similar chunks for each question, and sends them to Claude (claude-sonnet-5) with instructions to answer only from those sources and cite the source file for every fact. I first built it for Copperline Manufacturing's finance documents. To make it work with Larkfield Credit Union's documents, I added a --docs option to choose the document folder, a separate Chroma database for each folder (kept out of Git), a --save option that writes the retrieved chunks, a line containing only ANSWER, and Claude's answer to a text file, and I removed the Copperline-specific wording from Claude's instructions. My seven runs were Question 1 before and after fixing the duplicate fee schedule (q1-before.txt and q1-after.txt), then Questions 2 to 6 (q2-before.txt through q6-before.txt), covering a debit card fee, debit card fraud liability, a fee refund approval, check deposit availability, and a mortgage rate the documents do not cover. Along the way I also tightened Claude's instructions (no inferences beyond the sources, answer yes/no questions with Yes or No first, keep each condition with its own outcome, mention only conditions that could change the answer) and raised retrieval from 4 to 6 chunks, and saved extra reruns (q1-after_v2.txt, q4-after.txt, q5-after.txt) to compare.

## Why This Approach
Chunking method: structure

Makes sense on its own: Yes because each chunk is one complete section that starts with its document title and heading, for example "Funds Availability Policy (Version 3) - Longer holds" followed by the whole rule about seven-day holds and when they apply.

Label and source: Both because every chunk's text begins with "document title - section heading" and its metadata records the source file, which ask.py prints next to each retrieved chunk (for example, "[5] similarity 0.417  funds-availability-policy.html").

Short enough: Yes because I counted the words in every stored chunk: the longest is 137 words (median 52) in the Larkfield database and 137 words (median 58) in the Copperline database, all under about 190 words.

## Example Run
```text
[1] similarity 0.527  funds-availability-policy.html
Funds Availability Policy (Version 3) - Check deposits

For checks deposited at a branch or a Larkfield ATM, the first $275 of the day's check deposits is available on the next business day. The rest of the day's check deposits is available on the second business day after the day of deposit.

[2] similarity 0.500  funds-availability-policy.html
Funds Availability Policy (Version 3) - Mobile check deposits

Checks deposited with the Larkfield mobile app follow the same availability as checks deposited at a branch. A member can deposit up to $5,000 per business day through the mobile app. Members should keep the paper check for 30 days after a mobile deposit, then destroy it.

[3] similarity 0.432  card-dispute-procedure.html
Debit Card Dispute Procedure (Version 4) - How much the member may have to pay

A member's share of the loss from unauthorized debit card transactions depends on how quickly they report it. If the member reports a lost or stolen card within two business days of learning about it, they pay no more than $50. If they report it later than that, they may pay up to $500. If unauthorized transactions appear on a statement and the member does not report them within 60 days after the statement was sent, they may have to pay for transactions after those 60 days in full.

[4] similarity 0.425  funds-availability-policy.html
Funds Availability Policy (Version 3) - Business days and cutoff times

A business day is Monday through Friday, except federal holidays. A deposit made before 5 p.m. on a business day is received that day. A deposit made after 5 p.m., or on a weekend or holiday, is received on the next business day.

[5] similarity 0.417  funds-availability-policy.html
Funds Availability Policy (Version 3) - Longer holds

Larkfield may hold check deposits for up to seven business days when the day's check deposits total more than $6,000, when the account has been open fewer than 30 days, when the account has been overdrawn repeatedly in the last six months, or when there is reason to doubt the check will be paid. The member is told about a longer hold when the deposit is made, or by mail if the hold is placed later.

[6] similarity 0.413  consumer-fee-schedule.html
Consumer Fee Schedule (Version 8) - Official checks and money orders

A cashier's check costs $8. A money order costs $3. Members can buy up to five money orders in one day.

ANSWER
**$275** of the $2,000 check would be available on the next business day, with the remaining $1,725 available on the second business day after the deposit [funds-availability-policy.html].

A few details the question doesn't specify could change this:

- **Time of deposit**: This assumes the deposit was made before 5 p.m. on a business day, which counts as being received that day. If it was made after 5 p.m., or on a weekend/holiday, it would be treated as received the next business day, shifting the "next business day" availability forward accordingly [funds-availability-policy.html].
- **Longer holds**: If the member's total check deposits that day exceeded $6,000, if the account has been open fewer than 30 days, if the account has been overdrawn repeatedly in the last six months, or if there is reason to doubt the check will be paid, Larkfield may hold the funds for up to seven business days instead of the standard schedule above [funds-availability-policy.html]. The question doesn't indicate any of these conditions apply, so the standard $275 next-business-day availability is assumed.
```

## Reflection
What was wrong: consumer-fee-schedule-v7.html was an outdated Version 7 fee schedule still labeled "current" next to Version 8, and the retrieved chunks for Question 1 showed both versions' overdraft sections ranked first and second ($29 with a three-per-day limit versus $35 with a four-per-day limit) and both versions' monthly fee sections in third and fourth, so outdated copies took half the slots.

How I fixed it: Moved consumer-fee-schedule-v7.html out of week-03/larkfield-docs into a separate week-03/larkfield-docs-archive folder that ask.py does not index, then rebuilt the Larkfield database with --rebuild, which dropped it from 39 to 30 chunks so only Version 8 is retrieved.

Prevention: Archive process means that whenever Larkfield publishes a new version of a document, the document's owner moves the replaced version into an archive location outside the library the assistant indexes and marks it superseded, before the database is rebuilt.

Question 6: Declined because retrieval returned only unrelated chunks about stop payments, checking and overdraft fees, and share certificate rates and terms, all with similarity scores of 0.25 or lower, and none mentioned mortgages.

---

*AI disclosure: Claude Code (Anthropic's Claude Opus 5.5) helped me structure this writeup based on our iterative testing and discussions of the runs above.*
