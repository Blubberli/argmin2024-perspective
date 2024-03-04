# Readme Shared Task *Perspective Argument Retrieval*

This readme gives you all the important information for participation. The goal is to retrieve a set of best-matching
arguments for a given political aspect formulated as question.

## Instructions

The required packages are listed in the `requirements.txt` file. You can install them by running the following command:

```bash
pip install -r requirements.txt
```

The dataset is stored in a zip file, called `data-release-v1.5.zip`. Download the file and unzip it by running the
following command:

```bash
unzip data-release-v1.5.zip
```

### Scenarios

We distinguish between three scenarios for this retrieval task. Using these different scenarios, we want to verify the
effect of using socio-cultural properties at various stages. To participate, please follow the corresponding
instructions:

1.) **Baseline**: This scenario focuses on text-only retrieval of relevant arguments given and evaluates the general
abilities of the retrieval system. **Note**: do not use any socio-cultural properties for the query or the
corpus.

**Example query**:  _Are you in favor of the introduction of a tax on foods containing sugar (sugar tax)?_
**Example candidate**:  _The reduction of sugar in food should be pushed. Not every food needs additional sugar as a
supplement._

2.) **Explicit Perspectivism**: With this scenario, we focus on using explicitly mentioned socio-cultural properties
in the query and the corpus. **Note**: thus, you are allowed to integrate these properties for all queries and all
arguments in the corpus for retrieval.

**Example query**:

- **Text**: _Are you in favor of the introduction of a tax on foods containing sugar (sugar tax)?_
- **Age**: 18-34

**Example candidate**:

- **Text**: _Reducing sugar in food should be pushed. Not every food needs additional sugar as a supplement._
- **Age**: 18-34

3.) **Implicit Perspectivism**: With this scenario, we test the ability of a retrieval system to account for latently
encoded socio-cultural properties within the argument. **Note**: you are only allowed to use these properties for the
query **_not_** for the corpus.

**Example query**:

- **Text**: _Are you in favor of the introduction of a tax on foods containing sugar (sugar tax)?_
- **Age**: 18-34

**Example candidate**:  _The reduction of sugar in food should be pushed. Not every food needs additional sugar as a
supplement._

### Evaluation

We will evaluate the retrieval performance based on two core dimensions:

- **Relevance**: We will evaluate the relevance of the retrieved arguments to the given query. This quantifies the
  ability of the retrieval system to retrieve relevant arguments for a given question for scenario 1 or to retrieve
  relevant arguments for a given question and socio-cultural properties for scenarios 2 and 3.
    - ***nDCG***: Normalized Discounted Cumulative Gain (nDCG): this metric quantifies the quality of the ranking by
      putting more weight on the top-ranked arguments since it is more important to retrieve relevant arguments at
      lower ranks.
    - ***P@k***: Precision at k (P@k): this metric quantifies how many of the top-k retrieved arguments are relevant.
- **Diversity**: We will evaluate the fairness of the retrieval system by considering to what extent the ranking
  represents a diverse set of socio-cultural properties and whether minority groups are represented in the top-k
  retrieved arguments. Note that fairness for each query will be evaluated based on all socio-cultural properties that
  are not part of the query. The metrics will be averaged across all variables.
    - ***alpha-nDCG***: Alpha-nDCG: this metric works like nDCG but, on top of that, penalizes top-ranked items if they
      are not diverse. As a consequence, the metric rewards rankings that represent all relevant different socio-cultural
      properties at the top of the ranking.
    - ***rKL***: normalized discounted Kulback-Leibler divergence (rKL): this metric quantifies fairness independent of
      relevance. It measures whether the top-k retrieved arguments represent the minority groups of specific
      socio-cultural variables in the corpus.

To evaluate your systems, you must dump the predictions and then run the evaluation script on these predictions. 
`baseline.ipynb` shows these steps using `sentence-transformers` and `BM25` as baseline retrieval methods. 
It also shows you how to create the prediction file. 
With this `.jsonl` file, you must provide the corresponding best-matching candidates for each query as a JSON entry.
These entries must look as follows and include to keys `query_id` (id of the corresponding query) and `relevant candidates`, sorted list of the most relevant candidates.


```json
{
  "query_id":0,
  "relevant_candidates":[2019017914,201904055,201908061,201903763,...]
}
```
After dumping the results, you can run the evaluation with the following command:
```bash
python scripts/evaluation.py --data <path_to_corpus.jsonl> \
  --predictions <path_to_predictions.jsonl> \
  --output_dir <path_to_store_results> \
  --diversity True --scenario <baseline or perspective>  --split <train or dev>
```

You can evaluate your predictions as often as you'd like to. You only need to upload the prediction file for the official evaluation run. We will then run the script on the results of the unseen test data.
We will have two evaluations, one for relevance and one for diversity. We will report all four metrics across 4 different k values in both cases. We focus on nDCG and
alpha-nDCG as the main metrics for ranking participants.

## Baseline

The baseline is a simple SBERT-based retrieval system. It uses the pre-trained SBERT model from the
sentence-transformers
library to encode the queries and the arguments. The retrieval is then performed by computing the cosine similarity
between the query and the arguments. The top-k arguments are then returned as the retrieval results. There is no
training
involved in this baseline system.

## Data

This shared task is grounded on the x-stance dataset (Vamvas & Sennrich, 2020), providing arguments annotated with their
stance regarding different political issues gathered from the voting recommendation platform https://www.smartvote.ch/.
This platform provides voting suggestions based on a questionnaire that politicians and voters fill out. Therein,
politicians can argue why they are in favor or against specific political issues.

We use the arguments covering the 2019 Swiss Federal elections as a corpus and the political issues as queries.
Afterward, we enrich these arguments with eight political and demographic properties, either provided by the voting
platform itself (gender, age, party, …) or derived from the filled-out questionnaire of the politicians (political
attitude, important political aspects, …). This collection encapsulates 26,335 arguments for 45 political aspects from
German, French, and Italian.

We generate the train and development splits by considering 35 political aspects for training and 10 for development,
while the argument corpus is used for both sets. Apart from the queries for the baseline scenario, we will also provide
queries for the perspectivism scenarios, including socio-cultural information. As the x-stance dataset is publicly
available, the final evaluation data consists of secret test sets.

### Socio-Demographic Properties

We describe the socio-cultural profile of an author using the properties given by smartvote.ch. This includes eight
personal properties: gender, age, residence, education, civil status, denomination, political attitude, and a list of
important political issues, covering: open foreign policy, liberal economic policy, restrictive financial policy, law &
order, restrictive migration policy, and expanded environmental protection.

### Dataformat

We will provide the data in JSON files. First, the file corpus.jsonl, which consists of a collection of arguments and
the authors' socio-cultural profiles. Secondly, the queries without (baseline-queries) and with socio-cultural
properties (perspective-queries). Here are examples of what these JSON files will look like:

Example corpus entry `corpus.jsonl`:

    [{
    	"argument_id": "<argument_id>",
    	"text": "Eating is an individual decision. It doesn't need a nanny state.",
    	"target": "Are you in favor of the introduction of a tax on foods containing sugar …",
    	"stance": "CON",
    	"demographic_profile": {...}
    },…]

Example baseline query `baseline-queries/queries_train.jsonl`:

    [{
    	"query_id": "<query_id>",
    	"text": "Are you in favor of the introduction of a tax on foods containing sugar …",
    	"relevant_candidates": [23, 4623, 65, 321, ...]
    },…]

Example perspective query `perspective-queries/queries_train.jsonl`:

    [{
    	"query_id": "<query_id>",
    	"text": "Are you in favor of the introduction of a tax on foods containing sugar …",
    	"demographic_properties": {
    		"age": "18-34"
    	},
    	"relevant_candidates": [23, 4623, 65, 321, ...]
    },…]

### Submission

For the submission you will have to upload a JSON file with the following format:

    [{
    	"query_id": "<query_id>",
    	"retrieved_candidates": [23, 4623, 65, 321, ...]
    },…]    

This should contain each query_id from the test set and the predicted candidates as a ranked list from the full corpus.
Please retrieve the top-1000 candidates for each query. The evaluation script will then evaluate the relevance and
diversity of the retrieved candidates (at k = 4, 8, 16 and 20).

You can submit predictions for each scenario (baseline, explicit perspectivism, implicit perspectivism) or you can
only choose to submit for one or two scenarios of your choice. We will evaluate the submissions for each scenario
separately.

The submissions should be uploaded to your own submission folder which we will provide you with, once registered for
the shared task. You can upload your submission as often as you'd like to until the deadline. We will have two evaluation
runs before the final evaluation. You can submit your predictions for the official evaluation as often as you'd like to
until the deadline. We will consider those submissions that are uploaded at the time of the official evaluation, 
11.59 pm UTC -12h (“anywhere on Earth”) for each deadline. 


  
