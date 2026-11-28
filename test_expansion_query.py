
from query_expansion import expand_query

query = "Which tools did I use to build my chatbot?"

print("\nOriginal Query:")
print(query)

print("\nGenerating Expanded Queries...")

expanded_queries = expand_query(query)

print("\nExpanded Queries:")

for idx, expanded_query in enumerate(expanded_queries, start=1):
    print(f"{idx}. {expanded_query}")

assert len(expanded_queries) == 2, "Expected exactly 2 expanded queries"

assert all(
    isinstance(q, str) and q.strip()
    for q in expanded_queries
), "Invalid expanded query found"

print("\nQuery Expansion Test Passed!")
