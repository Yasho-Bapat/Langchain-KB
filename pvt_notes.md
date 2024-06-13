# Private notes
## Presentation
### slide 1 (recap)
So last week, we decided to use langchain for the knowledge base. We began talking about the chunking strategies, but we hit some roadblocks. We need to identify an appropriate chunking method for our use case.

### slide 2 (solution overview)
To do that, we conducted an experiment, where we implemented recursive and semantic splitters in langchain to compare their results. We also used the current pre-production version of the VAI knowledge base and saw how the answers it's providing are holding up against these two, since that one uses fixed-size chunking. <br> 
So to summarize the parameters we've used, we're using a 2000 character chunk size with 600 char overlap for recursive splitting. For semantic chunking, we're using the interquartile range as a breakpoint. <br> What that means is we're using the interquartile range of the distances between consecutive sentence groups to decide which sentence group is an outlier. A new chunk starts at an outlier. <br>
> this outlier is calculated by mean + factor*iqr

We've used OpenAI's text-embedding-3-large for the embedding function, along with PG Vector as our vector store. <br> To set up our experiment, we've ingested 10 MSDS (Material Safety Data Sheets) using these splitting strategies, and we've used GPT 3.5 Turbo as our LLM for answer generation. This is because GPT-4o and 4 does not support chat completion yet. 
Provide a risk assessment for handling and storing large quantities of Copper Sulphate.

### slide 3 (questions)
For our testbed, we created 30 questions and divided them into 3 categories, which we've defined here. Questions that involve only simple fact-based retrieval are in the easy category. <br> Questions which involve querying different parts of the same datasheet are in the moderate category and questions which involve querying different datasheets are in the hard category. <br> Then to have some quantitative data for these, we rated the answers and compared them to the actual answers from the MSDS documents. For that, we wrote the ground truths which we found from the datasheets. 

### slide 4 (results)
We can see the results summary in this table. Recursive does marginally better than semantic in easy, and better in the moderate category. But semantic does better in the hard category than both the chunkers. It has the most consistent performance across all 3 types of questions, which reflects in its overall average score. <br>  
We can show you the comprehensive results in our Google Sheet. 

### slide 5

### slide 7 (semchunk explanation)
This is a visual explanation of how the chunking works. To answer last time's question, here's a picture about how the different chunks are created. We are creating a new chunk wherever we see a spike in the distances between two consecutive sentences, as we understand them to be dissimilar in meaning. We use statistical methods of finding outliers in the spread of data. <br>
So what's happening in our implementation is:
- the sentence groups are embedded, and the distances between consecutive sentences are found.
- then, the interquartile range is found. That's done by defining Q1 as the 25th percentile, Q3 as the 75th percentile and then finding the difference between the two. We say that most of our datapoints lie within this range, and it is our non-problematic range. 
- To find the outlier, we define a fence, by ```mean + factor*iqr```. Any value in our distances which is more than this breakpoint value will be used to identify a new chunk. 
- We can see it in this picture 