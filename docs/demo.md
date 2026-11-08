# Demo

Real screenshots of the chat page (`public/`) talking to the API, taken against the 2,000-product catalog.

## Ask for a product and refine it

The assistant answers with the product title, the id and why it matches, and shows a card per product. The second question ("¿Y para pelo graso?") has no product in it: the rewriter uses the history to turn it into a full search.

![Search and follow-up](assets/screens/chat-2-followup.png)

## Out of scope questions are refused

Price, stock, delivery and returns are not in the catalog, so the topic classifier refuses them. Prompt injection gets the same fixed refusal and never reaches the generator.

![Refusals](assets/screens/chat-4-injection.png)

## Nothing in the catalog

The catalog is a sample, so some searches have no match. The assistant says so instead of inventing a product.

![Empty chat page](assets/screens/chat-0-empty.png)

> Known limits of the page: the answer is shown as plain text (Markdown such as `**bold**` is not rendered) and the intermediate step ("classify...", "retrieve...") is shown while waiting.
