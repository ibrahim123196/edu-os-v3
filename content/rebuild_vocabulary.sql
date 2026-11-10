-- One-off: build the word list from passages already indexed.
-- Normally ingestion does this, but a flush-ordering bug meant it ran
-- against an empty table. This fixes existing data without re-uploading.

TRUNCATE corpus_terms;

INSERT INTO corpus_terms (term, frequency)
SELECT word, COUNT(*)::int
FROM (
    SELECT regexp_split_to_table(lower(content), '[^a-z]+') AS word
    FROM document_chunks
) AS words
WHERE length(word) >= 4
GROUP BY word;

-- Check it worked: expect a few thousand rows, and coulomb should appear.
SELECT COUNT(*) AS total_terms FROM corpus_terms;
SELECT term, frequency FROM corpus_terms WHERE term LIKE 'coulo%' ORDER BY frequency DESC;
