# Readme Shared Task *Perspective Argument Retrieval*

This readme gives you all the important information for participate. The goal is to retrieve a set of best-matching
arguments for a given political aspect formulated as question.

## Instructions

The required packages are listed in the `requirements.txt` file. You can install them by running the following command:

```bash
pip install -r requirements.txt
```

The dataset is stored in a zip file, called `data-release-v1.5.zip`. Download the file and unzip it by running the
following command:

```bash
unzip data.zip
```

### Scenarios

We distinguish between three scenarios for this retrieval task. Using these different scenarios, we want to verify the
effect of using socio-cultural properties at varios stages. To participate, please follow the corresponding
instructions:

1.) **Baseline**: This scenario focus on text-only retrieval of relevant arguments given and evaluate the general
abilities of the retrieval system. **Note**: do not use any socio-cultural properties neither for the query nor the
cropus.

**Example query**:  _Are you in favor of the introduction of a tax on foods containing sugar (sugar tax)?_
**Example candidate**:  _The reduction of sugar in food should be pushed. Not every food needs additional sugar as a
supplement._

2.) **Explicit Perspectivism**: With this scenarion, we focusing on using explicit mentioned socio-cultural properties
in the query and the corpus. **Note**: thus you are allowed to integrate these properties for all queries and all
arguments in the corpus for retrieval.

**Example query**:

- **Text**: _Are you in favor of the introduction of a tax on foods containing sugar (sugar tax)?_
- **Age**: 18-34

**Example candidate**:

- **Text**: _Reducing sugar in food should be pushed. Not every food needs additional sugar as a supplement._
- **Age**: 18-34

3.) **Implicit Perspectivism**: With this scenarion, we test the ability of a retrieval system to account for latently
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
  relevant arguments for a given question and socio-cultural properties for scenario 2 and 3.
    - ***nDCG***: Normalized Discounted Cumulative Gain (nDCG): this metric quantifies the quality of the ranking by
      putting more weight on the top-ranked arguments, since it is more important to retrieve relevant arguments at
      lower ranks.
    - ***P@k***: Precision at k (P@k): this metric quantifies how many of the top-k retrieved arguments are relevant.
- **Fairness**: We will evaluate the fairness of the retrieval system by considering to what extent the ranking
  represents a diverse set of socio-cultural properties and whether minority groups are represented in the top-k
  retrieved arguments.
    - ***alpha-nDCG***: Alpha-nDCG: this metric works like nDCG but on top of that penalizes top-ranked items if they
      are not diverse. As a consequence the metric rewards rankins that represent all relevant different socio-cultural
      properties at the top of the ranking.
    - ***rKL***: normalized discounted Kulback-Leibler divergence (rKL): this metric quantifies fairness independent of
      relevance. It measures whether the top-k retrieved arguments are representative of the minority groups of specific
      socio-cultural variables in the corpus.

The evaluation script can be run by the following command:

```bash
python evaluation.py --data <path_to_corpus.jsonl> --predictions <path_to_predictions.jsonl> --output_dir <path_to_store_results> --diversity True
```

You can evaluate your predictions as often as you'd like to. For the official evaluation run, the script will be run on
the results
for the unseen test data. We will have two separate evaluations, one for the relevance and one for the fairness. In both
cases we will rank participants based on the average of the two metrics and the average across the different k-values.

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
German, French, and Italian. For simplicity, we provide examples translated into English in the following.

We generate the train and development splits by considering 35 political aspects for training and 10 for development,
while the argument corpus is used for both sets. Apart from the queries for the baseline scenario, we will also provide
queries for the perspectivism scenarios, including socio-cultural information. As the x-stance dataset is publicly
available, final evaluation data consist of secret test sets.

### Socio-Demographic Properties

We describe the socio-cultural profile of an author using the properties given by smartvote.ch. This includes eight
personal properties: gender, age, residence, education, civil status, denomination, political attitude, and a list of
important political issues, covering: open foreign policy, liberal economic policy, restrictive financial policy, law &
order, restrictive migration policy, expanded environmental protection.

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



  
