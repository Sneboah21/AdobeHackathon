METHODOLOGY 

Docker File implementation steps:
docker build --platform linux/amd64 -t adobe-hackathon-solution .
docker run -it -v "${PWD}\input:/app/input" -v "${PWD}\output:/app/output" adobe-hackathon-solution

Introduction :

A Python-based program called the Adobe Hackathon document analysis solution was created to automatically extract and analyze data from PDF documents. It tackles two main issues: first, it finds headings and subheadings to extract the documents' structural outline; second, it offers persona-driven content insights that are suited to particular user roles and tasks. The solution offers structured JSON formats for results and supports flexible execution, enabling users to run one or both analysis rounds.

Environment and Setup:

Python 3.7 or higher (ideally 3.10+) is needed for the solution, which runs in a separate virtual environment with pip installed for the installation of key dependencies, including scikit-learn for data processing, spaCy with the English language model for natural language processing, PyMuPDF for PDF parsing, and optionally psutil for system monitoring. The script supports local and Docker environments by automatically adjusting paths. Input PDFs and a persona configuration file (for Round 1B) should be placed in designated input directories, with matching output folders for JSON results

. In order to guarantee seamless document analysis workflows appropriate for hackathon conditions, users start the script, choose processing rounds from a menu, and take advantage of strong error handling, performance reporting, and clear logs.

Approach:

Heading Extractor: Working Explanation
The extractor uses PyMuPDF (fitz) to open the PDF file. This library allows access not just to plain text but also detailed layout and font metadata (e.g., font size, font name/style, text positions).
The extractor retrieves the text as a set of text blocks or spans for every page. Each span has characteristics like: 

The precise textual content
Font size (number value) 
Font family or style 
Coordinates of text position 

Since headings typically have larger sizes or consistent, distinctive fonts, this granular data is necessary to differentiate them from regular body text.
The extractor infers hierarchy levels. It may:
•	Sort unique font sizes 
•	Assign level 1 (top-level heading) 
•	Assign subsequent heading levels  progressively 

It then groups these extracted headings into a nested outline (tree structure) by their levels, maintaining the sequence and document flow.\

The created outline is then serialized to a hierarchical JSON format, with each heading containing its text, level, and potentially page number or position. After that, this output can be passed to subsequent processes or saved.

user-friendly interface:

In order to provide customized insights that are in line with the designated persona and their goals, this pipeline integrates document segmentation, natural language processing, and machine learning.


The options in the code relate to the interactive menu that the user sees during runtime, which lets them choose which part or parts of the document analysis solution to execute. Whether the script runs depends on these decisions: 

Round 1A Only: Only the document structure extraction phase is executed, during which the script analyzes PDFs to extract outlines and hierarchical headings.


Only Round 1B: Performs persona-driven content analysis, which analyzes up to ten PDFs for pertinent content summarization using persona and job-specific configuration. 

Both Rounds: Completes a comprehensive analysis workflow from structure extraction to persona-tailored intelligence in order, starting with Round 1A and ending with Round 1B. 

Quit: Closes the application without executing any further operations.




Persona Analyzer:

Load Persona Config: Reads persona_config.json containing persona and job-to-be-done strings driving personalized analysis.

Up to ten PDFs can be opened for input processing, and text divided into pages or logical sections can be extracted. 

Documents are divided into sections or subsections for focused analysis through section segmentation. 

NLP Processing: Tokenizes, identifies entities, and extracts keywords from each section using spaCy. 

Relevance Filtering: Evaluates sections for relevance by matching job/persona keywords, filtering content using ML models (scikit-learn), or similarity metrics. 

Summarization: Uses extractive summarization or clustering to produce succinct summaries of pertinent sections. 

Output: Saves metadata and persona-focused summaries in JSON files. 

Performance & Error Handling: Robust error catching and logging, with an analysis time of less than 60 seconds.






