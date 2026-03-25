# Word2Vec Hyperparameter Comparison

## Experiment Setup

- Query words: research, student, phd, exam
- Analogy tasks: ug:btech:pg; computer:science:mechanical; student:campus:faculty

## Best Checkpoints (Compact)

| Model | Checkpoint | Embedding Dim | Window Size | Negative Samples | Vocab Size | Query Coverage | Mean Top-1 Neighbor Cosine | Mean Top-1 Analogy Cosine | Combined Score | Missing Query Words |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| CBOW | cbow_model_2_100_10.pth | 100 | 2 | 10 | 2696 | 3/4 | 0.4420 | 0.3745 | 0.4082 | exam |
| SGNS | skipgram_model_5_100_10.pth | 100 | 5 | 10 | 941 | 3/4 | 0.5088 | 0.4395 | 0.4742 | exam |
| SGNS | skipgram_model_5_100_10.pth | 100 | 5 | 10 | 941 | 3/4 | 0.5088 | 0.4395 | 0.4742 | exam |

## Task-3: Nearest Neighbors and Analogies

### CBOW - cbow_model_2_100_10.pth

**Top-5 Nearest Neighbors**

- research: miscellaneous (0.4491), im (0.4019), pressing (0.3939), responsibilities (0.3795), lca (0.3635)
- student: events (0.4433), publications (0.3934), past (0.3743), affiliated (0.3621), inception (0.3411)
- phd: structural (0.4336), want (0.4148), lists (0.3824), doctorial (0.3792), sta (0.3791)
- exam: No results

**Analogy Results**

- ug:btech::pg:? -> foster (0.3727), agreements (0.3555), fortified (0.3514), tentative (0.3506), contacts (0.3504)
- computer:science::mechanical:? -> mlops (0.4184), chemical (0.3828), electrical (0.3625), foundational (0.3255), undergraduates (0.3191)
- student:campus::faculty:? -> responsibility (0.3325), etl (0.2954), instrumentation (0.2822), undergone (0.2741), initiation (0.2645)

### SGNS - skipgram_model_5_100_10.pth

**Top-5 Nearest Neighbors**

- research: central (0.5935), image (0.5442), xrf (0.4885), activated (0.4669), enabled (0.4606)
- student: oversight (0.4147), fee (0.4072), structure (0.3598), affairs (0.3595), academics (0.3572)
- phd: process (0.5183), management (0.4848), ay (0.4742), structure (0.4668), time (0.4655)
- exam: No results

**Analogy Results**

- ug:btech::pg:? -> intelligent (0.4580), mtech (0.4482), mar (0.4207), computational (0.4098), related (0.4076)
- computer:science::mechanical:? -> meter (0.4251), complaint (0.4189), future (0.4074), contact (0.4036), simulink (0.4003)
- student:campus::faculty:? -> hal (0.4354), solar (0.4053), co (0.3986), guest (0.3924), list (0.3886)

### SGNS - skipgram_model_5_100_10.pth

**Top-5 Nearest Neighbors**

- research: central (0.5935), image (0.5442), xrf (0.4885), activated (0.4669), enabled (0.4606)
- student: oversight (0.4147), fee (0.4072), structure (0.3598), affairs (0.3595), academics (0.3572)
- phd: process (0.5183), management (0.4848), ay (0.4742), structure (0.4668), time (0.4655)
- exam: No results

**Analogy Results**

- ug:btech::pg:? -> intelligent (0.4580), mtech (0.4482), mar (0.4207), computational (0.4098), related (0.4076)
- computer:science::mechanical:? -> meter (0.4251), complaint (0.4189), future (0.4074), contact (0.4036), simulink (0.4003)
- student:campus::faculty:? -> hal (0.4354), solar (0.4053), co (0.3986), guest (0.3924), list (0.3886)

## Notes

- Higher cosine values usually mean stronger semantic alignment.
- NA means that setting was not found in file naming.
- Combined Score = (NN@1(avg) + ANA@1(avg)) / 2
