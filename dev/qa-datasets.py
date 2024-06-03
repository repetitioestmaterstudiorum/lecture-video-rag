tuning_dataset = [
    #
    #
    # --- CS50's Introduction to Artificial Intelligence with Python 2023
    #
    #
    {
        'course': "CS50's Introduction to Artificial Intelligence with Python 2023",
        'video': "2 - Search - Lecture 0",
        'question': """Between depth first search (DFS) and breadth first search (BFS), which will find a shorter path through a maze?
1) DFS will always find a shorter path than BFS
2) BFS will always find a shorter path than DFS
3) DFS will sometimes, but not always, find a shorter path than BFS
4) BFS will sometimes, but not always, find a shorter path than DFS
5) Both algorithms will always find paths of the same length""",
        'ground_truth': "4) BFS will sometimes, but not always, find a shorter path than DFS"
    },
    {
        'course': "CS50's Introduction to Artificial Intelligence with Python 2023",
        'video': "2 - Search - Lecture 0",
        'question': """Why is depth-limited minimax sometimes preferable to minimax without a depth limit?
1) Depth-limited minimax can arrive at a decision more quickly because it explores fewer states
2) Depth-limited minimax will achieve the same output as minimax without a depth limit, but can sometimes use less memory
3) Depth-limited minimax can make a more optimal decision by not exploring states known to be suboptimal
4) Depth-limited minimax is never preferable to minimax without a depth limit""",
        'ground_truth': "1) Depth-limited minimax can arrive at a decision more quickly because it explores fewer states"
    },
    {
        'course': "CS50's Introduction to Artificial Intelligence with Python 2023",
        'video': "3 - Knowledge - Lecture 1",
        'question': """Consider these logical sentences:

If Hermione is in the library, then Harry is in the library.
Hermione is in the library.
Ron is in the library and Ron is not in the library.
Harry is in the library.
Harry is not in the library or Hermione is in the library.
Ron is in the library or Hermione is in the library.
Which of the following logical entailments is true?

a) Sentence 6 entails Sentence 2
b) Sentence 1 entails Sentence 4
c) Sentence 6 entails Sentence 3
d) Sentence 2 entails Sentence 5
e) Sentence 1 entails Sentence 2
f) Sentence 5 entails Sentence 6""",
        'ground_truth': "d) Sentence 2 entails Sentence 5"
    },
    {
        'course': "CS50's Introduction to Artificial Intelligence with Python 2023",
        'video': "3 - Knowledge - Lecture 1",
        'question': """Let propositional variable R be that “It is raining,” the variable C be that “It is cloudy,” and the variable S be that “It is sunny.” Which of the following a propositional logic representation of the sentence “If it is raining, then it is cloudy and not sunny.”?

1) (R → C) ∧ ¬S
2) R → C → ¬S
3) R ∧ C ∧ ¬S
4) R → (C ∧ ¬S)
5) (C v ¬S) → R""",
        'ground_truth': "4) R → (C ∧ ¬S)"
    },
    {
        'course': "CS50's Introduction to Artificial Intelligence with Python 2023",
        'video': "3 - Knowledge - Lecture 1",
        'question': """Consider, in first-order logic, the following predicate symbols. Student(x) represents the predicate that “x is a student.” Course(x) represents the predicate that “x is a course.” Enrolled(x, y) represents the predicate that “x is enrolled in y.” Which of the following is a first-order logic translation of the sentence “There is a course that Harry and Hermione are both enrolled in.”?

1) ∃x. Course(x) ∧ Enrolled(Harry, x) ∧ Enrolled(Hermione, x)
2) ∀x. Course(x) ∧ Enrolled(Harry, x) ∧ Enrolled(Hermione, x)
3) ∃x. Enrolled(Harry, x) ∧ ∃y. Enrolled(Hermione, y)
4) ∀x. Enrolled(Harry, x) ∧ ∀y. Enrolled(Hermione, y)
5) ∃x. Enrolled(Harry, x) v Enrolled(Hermione, x)
6) ∀x. Enrolled(Harry, x) v Enrolled(Hermione, x)
""",
        'ground_truth': "1) ∃x. Course(x) ∧ Enrolled(Harry, x) ∧ Enrolled(Hermione, x)"
    },
    {
        'course': "CS50's Introduction to Artificial Intelligence with Python 2023",
        'video': "4 - Uncertainty - Lecture 2",
        'question': """Consider a standard 52-card deck of cards with 13 card values (Ace, King, Queen, Jack, and 2-10) in each of the four suits (clubs, diamonds, hearts, spades). If a card is drawn at random, what is the probability that it is a spade or a two?

Note that “or” in this question refers to inclusive, not exclusive, or.

1) About 0.019
2) About 0.077
3) About 0.17
4) About 0.25
5) About 0.308
6) About 0.327
7) About 0.5
8) None of the above""",
        'ground_truth': "5) About 0.308"
    },
    {
        'course': "CS50's Introduction to Artificial Intelligence with Python 2023",
        'video': "4 - Uncertainty - Lecture 2",
        'question': """Imagine flipping two fair coins, where each coin has a Heads side and a Tails side, with Heads coming up 50% of the time and Tails coming up 50% of the time. What is probability that after flipping those two coins, one of them lands heads and the other lands tails?""",
        'ground_truth': "The probability is 0.5 (1/2)"
    },
    {
        'course': "CS50's Introduction to Artificial Intelligence with Python 2023",
        'video': "5 - Optimization - Lecture 3",
        'question': """For which of the following will you always find the same solution, even if you re-run the algorithm multiple times?

Assume a problem where the goal is to minimize a cost function, and every state in the state space has a different cost.

1) Steepest-ascent hill-climbing, each time starting from a different starting state
2) Steepest-ascent hill-climbing, each time starting from the same starting state
3) Stochastic hill-climbing, each time starting from a different starting state
4) Stochastic hill-climbing, each time starting from the same starting state
5) Both steepest-ascent and stochastic hill climbing, so long as you always start from the same starting state
6) Both steepest-ascent and stochastic hill climbing, each time starting from a different starting state
7) No version of hill-climbing will guarantee the same solution every time""",
        'ground_truth': "2) Steepest-ascent hill-climbing, each time starting from the same starting state"
    },
    {
        'course': "CS50's Introduction to Artificial Intelligence with Python 2023",
        'video': "5 - Optimization - Lecture 3",
        'question': """Consider this optimization problem:

A farmer is trying to plant two crops, Crop 1 and Crop 2, and wants to maximize his profits. The farmer will make $500 in profit from each acre of Crop 1 planted, and will make $400 in profit from each acre of Crop 2 planted.

However, the farmer needs to do all of his planting today, during the 12 hours between 7am and 7pm. Planting an acre of Crop 1 takes 3 hours, and planting an acre of Crop 2 takes 2 hours.

The farmer is also limited in terms of supplies: he has enough supplies to plant 10 acres of Crop 1 and enough supplies to plant 4 acres of Crop 2.

Assume the variable C1 represents the number of acres of Crop 1 to plant, and the variable C2 represents the number of acres of Crop 2 to plant.

What would be a valid objective function for this problem?

a) 500 * C1 + 400 * C2
b) 500 * 10 * C1 + 400 * 4 * C2
c) 10 * C1 + 4 * C2
d) -3 * C1 - 2 * C2
e) C1 + C2""",
        'ground_truth': "a) 500 * C1 + 400 * C2"
    },
    {
        'course': "CS50's Introduction to Artificial Intelligence with Python 2023",
        'video': "6 - Learning - Lecture 4",
        'question': """Categorize the following: A social network's AI uses existing tagged photos of people to identify when those people appear in new photos.

    1) This is an example of supervised learning
    2) This is an example of reinforcement learning
    3) This is an example of unsupervised learning
    4) This is not an example of machine learning""",
        'ground_truth': "1) This is an example of supervised learning"
    },
    {
        'course': "CS50's Introduction to Artificial Intelligence with Python 2023",
        'video': "6 - Learning - Lecture 4",
        'question': """Imagine a regression AI that makes the following predictions for the following 5 data points. What is the total L2 loss across all of these data points (i.e., the sum of all the individual L2 losses for each data point)?

1) The true output is 2 and the AI predicted 4.
2) The true output is 4 and the AI predicted 5.
3) The true output is 4 and the AI predicted 3.
4) The true output is 5 and the AI predicted 2.
5) The true output is 6 and the AI predicted 5.""",
        'ground_truth': "16"
    },
    {
        'course': "CS50's Introduction to Artificial Intelligence with Python 2023",
        'video': "6 - Learning - Lecture 4",
        'question': """If Hypothesis 1 has a lower L1 loss and a lower L2 loss than Hypothesis 2 on a set of training data, why might Hypothesis 2 still be a preferable hypothesis?

1) Hypothesis 1 might be the result of regularization.
2) Hypothesis 1 might be the result of overfitting.
3) Hypothesis 1 might be the result of loss.
4) Hypothesis 1 might be the result of cross-validation.
5) Hypothesis 1 might be the result of regression.""",
        'ground_truth': "2) Hypothesis 1 might be the result of overfitting."
    },
    {
        'course': "CS50's Introduction to Artificial Intelligence with Python 2023",
        'video': "6 - Learning - Lecture 4",
        'question': """In the ε-greedy approach to action selection in reinforcement learning, which of the following values of ε makes the approach identical to a purely greedy approach?

a) ε = 0
b) ε = 0.25
c) ε = 0.5
d) ε = 0.75
e) ε = 1""",
        'ground_truth': "a) ε = 0"
    },
    {
        'course': "CS50's Introduction to Artificial Intelligence with Python 2023",
        'video': "7 - Neural Networks - Lecture 5",
        'question': """Consider the below neural network, where we set:

w0 = -5
w1 = 2
w2 = -1 and
w3 = 3.
x1, x2, and x3 represent input neurons, and y represents the output neuron.

What value will this network compute for y given inputs x1 = 3, x2 = 2, and x3 = 4 if we use a step activation function? What if we use a ReLU activation function?

a) 0 for step activation function, 0 for ReLU activation function
b) 0 for step activation function, 1 for ReLU activation function
c) 1 for step activation function, 0 for ReLU activation function
d) 1 for step activation function, 1 for ReLU activation function
e) 1 for step activation function, 11 for ReLU activation function
f) 1 for step activation function, 16 for ReLU activation function
g) 11 for step activation function, 11 for ReLU activation function
h) 16 for step activation function, 16 for ReLU activation function""",
        'ground_truth': "e) 1 for step activation function, 11 for ReLU activation function"
    },
    {
        'course': "CS50's Introduction to Artificial Intelligence with Python 2023",
        'video': "7 - Neural Networks - Lecture 5",
        'question': """How many total weights (including biases) will there be for a fully connected neural network with a single input layer with 3 units, a single hidden layer with 5 units, and a single output layer with 4 units?""",
        'ground_truth': "44"
    },
    {
        'course': "CS50's Introduction to Artificial Intelligence with Python 2023",
        'video': "7 - Neural Networks - Lecture 5",
        'question': """Consider a recurrent neural network that listens to a audio speech sample, and classifies it according to whose voice it is. What network architecture is the best fit for this problem?

1) One-to-one (single input, single output)
2) Many-to-one (multiple inputs, single output)
3) One-to-many (single input, multiple outputs)
4) Many-to-many (multiple inputs, multiple outputs)""",
        'ground_truth': "2) Many-to-one (multiple inputs, single output)"
    },
    {
        'course': "CS50's Introduction to Artificial Intelligence with Python 2023",
        'video': "8 - Language - Lecture 6",
        'question': """Consider the below context-free grammar, where S is the start symbol.

S -> NP V
NP -> N | A NP
A -> "small" | "white"
N -> "cats" | "trees"
V -> "climb" | "run"

Consider also the following four sentences:
1) Cats run.
2) Cats climb trees.
3) Small cats run.
4) Small white cats climb.

Of the four sentences above, which sentences can be derived from the context-free grammar?""",
        'ground_truth': "Sentence 1, Sentence 3, and Sentence 4."
    },
    {
        'course': "CS50's Introduction to Artificial Intelligence with Python 2023",
        'video': "8 - Language - Lecture 6",
        'question': """Which of the following is not a true statement?

1) Attention mechanisms can be used to determine which parts of an input sequence are most important to focus on.
2) One-hot representations of words better represent word meaning than distributed representations of words.
3) Transformers can be faster to train than recurrent neural networks because they are more easily parallelized.
4) A Naive Bayes Classifier assumes that the order of words doesn’t matter when determining how they should be classified.""",
        'ground_truth': "2) One-hot representations of words better represent word meaning than distributed representations of words."
    },
    {
        'course': "CS50's Introduction to Artificial Intelligence with Python 2023",
        'video': "8 - Language - Lecture 6",
        'question': """Why is “smoothing” useful when applying Naive Bayes?

1) Smoothing allows Naive Bayes to better handle cases where evidence has never appeared for a particular category.
2) Smoothing allows Naive Bayes to better handle cases where there are many categories to classify between, instead of just two.
3) Smoothing allows Naive Bayes to be less “naive” by not assuming that evidence is conditionally independent.
4) Smoothing allows Naive Bayes to turn a conditional probability of evidence given a category into a probability of a category given evidence.""",
        'ground_truth': "1) Smoothing allows Naive Bayes to better handle cases where evidence has never appeared for a particular category."
    },
    {
        'course': "CS50's Introduction to Artificial Intelligence with Python 2023",
        'video': "8 - Language - Lecture 6",
        'question': """From the phrase “must be the truth”, how many word n-grams of length 2 can be extracted?""",
        'ground_truth': "3"
    },
    #
    #
    # --- MIT 9.00SC Introduction to Psychology, Fall 2011
    #
    #
    {
        'course': "MIT 9.00SC Introduction to Psychology, Fall 2011",
        'video': "",
        'question': """In a social psychology experiment, white participants either forecasted (imagined) or actually experienced how they would feel and how they would behave towards a white stranger who made a racial slur about a black stranger. The findings were:

A) both forecasters and experiencers felt negative feelings towards the white stranger and were
less likely to select the white stranger as a partner for performing a task (relative to a control
condition without any slur being made)
B) neither forecasters nor experiencers felt negative feelings towards the white stranger and
both groups were likely to select the white stranger as a partner for performing a task
(relative to a control condition without any slur being made)
C) forecasters felt negative feelings towards the white stranger and were less likely to select the
white stranger as a partner for performing the task; experiencers felt negative feelings
towards the white stranger but were as likely to select the white stranger as a white stranger
in a control condition without any slur being made
D) forecasters felt negative feelings towards the white stranger and were less likely to select the
white stranger as a partner for performing the task; experiencers did not have negative
feelings towards the white stranger and were as likely to select the white stranger as a white
stranger in a control condition without any slur being made""",
        'ground_truth': "D) forecasters felt negative feelings towards the white stranger and were less likely to select the white stranger as a partner for performing the task; experiencers did not have negative feelings towards the white stranger and were as likely to select the white stranger as a white stranger in a control condition without any slur being made"
    },
    {
        'course': "MIT 9.00SC Introduction to Psychology, Fall 2011",
        'video': "",
        'question': """Experiments have shown that which manipulations can enhance one's feelings or positive actions towards another person?
A) holding a warm drink relative to a cold drink just before evaluating another person
B) not thinking about money
C) rotating from table to table at a speed dating event
D) all of the above""",
        'ground_truth': "D) all of the above"
    },
    {
        'course': "MIT 9.00SC Introduction to Psychology, Fall 2011",
        'video': "",
        'question': """Empirical evidence supports which one conclusion?
A) hypnosis is effective in highly hypnotizable people
B) the Rorschach inkblot test is a valid clinical test
C) medical school interviews are effective for identifying which applicants will perform well in
medical school classes
D) praising a child's intelligence helps them perform well on a challenging task""",
        'ground_truth': "A) hypnosis is effective in highly hypnotizable people"
    },
    {
        'course': "MIT 9.00SC Introduction to Psychology, Fall 2011",
        'video': "",
        'question': """For a typical split-brain (= callosotomy) patient, if a picture of a pencil is presented in the right visual field and a picture of a hammer is presented to the left visual field and the patient is
asked to identify what had been presented
A) the patient would say “pencil” and would pick out a hammer from an array of unseen
objects with the left hand
B) the patient would say “hammer” and would pick out a pencil from an array of unseen
objects with the left hand
C) the patient would say “hammer” and would pick out a hammer from an array of unseen
objects with the left hand
D) the patient would say “pencil” and would pick out a pencil from an array of unseen objects
with the left hand""",
        'ground_truth': "A) the patient would say “pencil” and would pick out a hammer from an array of unseen objects with the left hand"
    },
    {
        'course': "MIT 9.00SC Introduction to Psychology, Fall 2011",
        'video': "",
        'question': """At what point in the human visual system are neurons or axons that code for the same visual
field from the two different eyes first brought together anatomically?
A) rods and cones
B) retina
C) optic chiasm
D) primary visual cortex""",
        'ground_truth': "C) optic chiasm"
    },
    {
        'course': "MIT 9.00SC Introduction to Psychology, Fall 2011",
        'video': "",
        'question': """The parvocellular visual pathway, relative to the magnocellular visual pathway,
A) has many more cells, is wavelength sensitive, and has slow, sustained responses
B) has many fewer cells, is wavelength insensitive, and has rapid, transient responses
C) has many fewer cells, is wavelength insensitive, and has slow, sustained responses
D) has many more cells, is wavelength sensitive, and has rapid, transient responses""",
        'ground_truth': "A) has many more cells, is wavelength sensitive, and has slow, sustained responses"
    },
    {
        'course': "MIT 9.00SC Introduction to Psychology, Fall 2011",
        'video': "",
        'question': """Which of these aspects of face perception is NOT true for infants?
A) human infants have better memory for faces from their family's racial group
B) human infants prefer top-heavy facial configurations
C) human infants have similar memory for human and monkey faces
D) monkey infants prefer seeing photographs of faces than photographs of objects the first time
they see a face""",
        'ground_truth': "A) human infants have better memory for faces from their family's racial group"
    },
    {
        'course': "MIT 9.00SC Introduction to Psychology, Fall 2011",
        'video': "",
        'question': """We read about
A) 2 letters at a time
B) 7 letters at a time
C) 12 letters at a time
D) 22 letters at a time""",
        'ground_truth': "C) 12 letters at a time"
    },
    {
        'course': "MIT 9.00SC Introduction to Psychology, Fall 2011",
        'video': "",
        'question': """A variety of results, including results from a spatial cuing task from Posner, indicate that
patients with spatial neglect due to right posterior lesions have a primary deficit in
A) engaging attention to the right field
B) engaging attention to the left field
C) disengaging attention from the left field
D) disengaging attention from the right field""",
        'ground_truth': "D) disengaging attention from the right field"
    },
    {
        'course': "MIT 9.00SC Introduction to Psychology, Fall 2011",
        'video': "",
        'question': """“Negative contrast” refers to the finding that rat performance in a maze is worst at the end of
the experiment for rats who
A) receive a large reward throughout the experiment
B) receive a small reward throughout the experiment
C) initially receive a large reward and then are switched to a small reward
D) initially receive a small reward and then are switched to a large reward""",
        'ground_truth': "C) initially receive a large reward and then are switched to a small reward"
    },
    {
        'course': "MIT 9.00SC Introduction to Psychology, Fall 2011",
        'video': "",
        'question': """Which finding or findings below do NOT challenge the claims of behaviorism that conditioning
could relate any conditioned stimulus to any conditioned response?
A) evidence that rats on their own would relate nausea to taste and shock to lights and sounds
B) evidence that Little Albert transferred learned fear conditioning to multiple animals but not
wooden blocks
C) evidence for latent learning in rats
D) evidence for the partial-reinforcement effect""",
        'ground_truth': "D) evidence for the partial-reinforcement effect"
    },
    {
        'course': "MIT 9.00SC Introduction to Psychology, Fall 2011",
        'video': "",
        'question': """The number of items that can be held in short-term memory is typically conceptualized as:

A) 3 plus or minus 2
B) 5 plus or minus 2
C) 7 plus or minus 2
D) 9 plus or minus 2""",
        'ground_truth': "C) 7 plus or minus 2"
    },
    {
        'course': "MIT 9.00SC Introduction to Psychology, Fall 2011",
        'video': "",
        'question': """Chess masters and chess beginners were shown chess pieces on a chessboard, and then asked to reconstruct the locations of the chess pieces from memory. Some of the pieces were shown from normal games (normal arrays) and some were shown in random arrays. Researchers found that:
A) Chess masters had superior memory relative to chess beginners for chess pieces in normal arrays, and the two groups had equal memory for chess pieces in random arrays.
B) Chess masters had superior memory relative to chess beginners for chess pieces in both normal and random arrays.
C) Chess masters had superior memory relative to chess beginners for chess pieces in random arrays, and the two groups had equal memory for chess pieces in normal arrays.
D) Chess masters had superior memory for chess pieces in normal arrays, and inferior memory relative to chess beginners for chess pieces in random arrays.""",
        'ground_truth': "D) Chess masters had superior memory for chess pieces in normal arrays, and inferior memory relative to chess beginners for chess pieces in random arrays."
    },
    {
        'course': "MIT 9.00SC Introduction to Psychology, Fall 2011",
        'video': "",
        'question': """A patient with a right-sided removal of the hippocampus would be impaired on which of the following?
A) short-term verbal memory
B) long-term verbal memory
C) short-term visuo-spatial memory
D) long-term visuo-spatial memory""",
        'ground_truth': "D) long-term visuo-spatial memory"
    },
    {
        'course': "MIT 9.00SC Introduction to Psychology, Fall 2011",
        'video': "",
        'question': """Patients with anterograde global amnesia typically have:
A) a temporally limited retrograde amnesia.
B) no retrograde amnesia at all.
C) a retrograde amnesia for the most distant past parts of their lives.
D) a complete retrograde amnesia.""",
        'ground_truth': "A) a temporally limited retrograde amnesia."
    },
    {
        'course': "MIT 9.00SC Introduction to Psychology, Fall 2011",
        'video': "",
        'question': """Experimental evidence indicates which of the following about lexical access (thinking about the meaning of words)?
A) All meanings of words are activated for about 500 msec, and then only the relevant meaning is activated at 2000 msec.
B) All meanings of words are activated for about 500 msec through 2000 msec.
C) Only relevant meanings of words are activated for about 500 msec through 2000 msec.
D) Relevant meanings of words are activated for about 500 msec, and then all meanings are activated at 2000 msec.""",
        'ground_truth': "A) All meanings of words are activated for about 500 msec, and then only the relevant meaning is activated at 2000 msec."
    },
    {
        'course': "MIT 9.00SC Introduction to Psychology, Fall 2011",
        'video': "",
        'question': """Which of the following is NOT true about human language development in children?
A) Left hemisphere specialization for speech is evident within days of birth.
B) Children can distinguish all sounds in all languages up to about an age of 3 years.
C) Girls, on average, learn the meanings of more words in the first two years of life.
D) Parental communication in “motherese” involves short pauses, careful enunciation, and exaggerated intonation in a high pitch that helps infants perceive language.""",
        'ground_truth': "B) Children can distinguish all sounds in all languages up to about an age of 3 years."
    },
    {
        'course': "MIT 9.00SC Introduction to Psychology, Fall 2011",
        'video': "",
        'question': """Patients with right or left hemisphere lesions were compared to healthy control subjects in their abilities to interpret (identify) people who are lying through facial expressions alone
or through facial expressions and vocal cues. What was found?
A) Patients with right hemisphere lesions were more accurate than patients with left hemisphere lesions and healthy people.
B) Patients with left hemisphere lesions were more accurate than patients with right hemisphere lesions and healthy people.
C) Patients with right hemisphere lesions were as accurate as controls and more accurate than patients with left hemisphere lesions.
D) Patients with left hemisphere lesions were as accurate as controls and more accurate than patients with right hemisphere lesions.""",
        'ground_truth': "B) Patients with left hemisphere lesions were more accurate than patients with right hemisphere lesions and healthy people."
    },
    {
        'course': "MIT 9.00SC Introduction to Psychology, Fall 2011",
        'video': "",
        'question': """Experimental studies show that, for equal losses or gains, people are:
A) risk averse for losses and gains.
B) risk taking for losses and gains.
C) risk averse for gains and risk taking for losses.
D) risk taking for gains and risk averse for losses.""",
        'ground_truth': "C) risk averse for gains and risk taking for losses."
    },
    {
        'course': "MIT 9.00SC Introduction to Psychology, Fall 2011",
        'video': "",
        'question': """The children of highly successful people are often less successful than their parents. This observation reflects most certainly
A) the pressure of growing up with enormous parental expectations.
B) confirmation bias.
C) regression to the mean.
D) the lack of attention from parents devoted to career.""",
        'ground_truth': "C) regression to the mean."
    },
    {
        'course': "MIT 9.00SC Introduction to Psychology, Fall 2011",
        'video': "",
        'question': """Which is NOT true about IQ scores according to available evidence?
A) IQ scores are steadily rising around the world.
B) Crystallized intelligence shows little decline in normal aging.
C) Fluid intelligence shows little decline in normal aging.
D) IQ scores, according to twin studies, are about 50% heritable.""",
        'ground_truth': "C) Fluid intelligence shows little decline in normal aging."
    },
    {
        'course': "MIT 9.00SC Introduction to Psychology, Fall 2011",
        'video': "",
        'question': """IQ measures predict or account for about what % of variation in outcomes such as school GPA, job success, and salary?
A) 5%
B) 25%
C) 50%
D) 75%
""",
        'ground_truth': "B) 25%"
    },
    {
        'course': "MIT 9.00SC Introduction to Psychology, Fall 2011",
        'video': "",
        'question': """Activation in which brain region has been most closely tied to performance on intelligence tests?
A) occipital lobe
B) temporal lobe
C) parietal lobe
D) frontal lobe""",
        'ground_truth': "D) frontal lobe"
    },
    {
        'course': "MIT 9.00SC Introduction to Psychology, Fall 2011",
        'video': "",
        'question': """Which of the following could be interpreted as support for the James-Lange hypothesis that a bodily response leads to a subjective experience of emotion?
A) Using pencils to force a smile or prevent a smile alters emotional experience.
B) Following instructions to move facial musculature into specific expressions enhances emotional experience consistent with that expression.
C) Men were more likely to call a woman they met in the middle of a dangerous bridge than a safe bridge.
D) All of the above.""",
        'ground_truth': "D) All of the above."
    },
    {
        'course': "MIT 9.00SC Introduction to Psychology, Fall 2011",
        'video': "",
        'question': """Lesions to the amygdala in humans result in all EXCEPT:
A) loss of fear conditioning as measured by autonomic (GSR) measures.
B) loss of emotional enhancement of memory.
C) loss of ability to identify fearful facial expressions.
D) loss of ability to identity disgust facial expressions.""",
        'ground_truth': "D) loss of ability to identity disgust facial expressions."
    },
    {
        'course': "MIT 9.00SC Introduction to Psychology, Fall 2011",
        'video': "",
        'question': """Imaging studies of the amygdala indicate all of the below EXCEPT:
A) selective response to fearful faces in subliminal presentations.
B) selective response to fearful faces in a cortically blind visual field.
C) greater amygdala responses to scenes judged as more negatively intense.
D) greater activation in women in the left amygdala as they rate the intensity of scenes and in the right amygdala as they form long-term memories for the scenes.""",
        'ground_truth': "D) greater activation in women in the left amygdala as they rate the intensity of scenes and in the right amygdala as they form long-term memories for the scenes."
    },
    {
        'course': "MIT 9.00SC Introduction to Psychology, Fall 2011",
        'video': "",
        'question': """Which is NOT true about measures of personality?
A) their stability increases with age
B) personality traits, according to twin studies, are about 50% heritable
C) conscientiousness appears to diminish risk of Alzheimer's disease
D) children are more similar to an adoptive sibling than to any randomly selected unrelated child""",
        'ground_truth': "D) children are more similar to an adoptive sibling than to any randomly selected unrelated child"
    },
    {
        'course': "MIT 9.00SC Introduction to Psychology, Fall 2011",
        'video': "",
        'question': """Research indicates that for a test taken in the afternoon
A) introverts and extraverts perform better after consuming caffeine
B) introverts and extraverts perform worse after consuming caffeine
C) introverts perform better but extraverts perform worse after consuming caffeine
D) extraverts perform better but introverts perform worse after consuming caffeine""",
        'ground_truth': "C) introverts perform better but extraverts perform worse after consuming caffeine"
    },
    {
        'course': "MIT 9.00SC Introduction to Psychology, Fall 2011",
        'video': "",
        'question': """In one study (from Woodward, 1998), 3-month olds and 6-month olds saw repeatedly a person reach for an object (ball) on the left and not reach for a teddy bear on the right. Then, they saw a display in which the teddy bear was on the left, and the ball on the right. The person either reached for the teddy bear on the left (same action, new object) or the ball on the right (same object, new action). Looking time was used as a measure to infer how the 3-month olds and 6-month olds interpreted the relation between the initial habituation phase and the subsequent changed test phases. The results indicated
A) both 3-month olds and 6-month olds looked longer when the person grabbed a new object than when the person made a new action
B) both 3-month olds and 6-month olds looked longer when the person made a new action than when the person grabbed a new object
C) 6-month olds looked longer when the person made a new action, but 3-month olds looked longer when the person grabbed a new object
D) 3-month olds looked longer when the person made a new action, but 6-month olds looked longer when the person grabbed a new object""",
        'ground_truth': "D) 3-month olds looked longer when the person made a new action, but 6-month olds looked longer when the person grabbed a new object"
    },
    {
        'course': "MIT 9.00SC Introduction to Psychology, Fall 2011",
        'video': "",
        'question': """Theory of mind research has reported which findings?
A) maturation of theory of mind occurs earlier in interdependent than independent cultures, earlier in children with older siblings, and later in autism
B) maturation of theory of mind occurs earlier in interdependent than independent cultures, earlier in children with younger siblings, and later in autism
C) maturation of theory of mind occurs similarly in interdependent and independent cultures, earlier in children with older siblings, and later in autism
D) maturation of theory of mind occurs similarly in interdependent and independent cultures, earlier in children with younger siblings, and later in autism""",
        'ground_truth': "C) maturation of theory of mind occurs similarly in interdependent and independent cultures, earlier in children with older siblings, and later in autism"
    },
    {
        'course': "MIT 9.00SC Introduction to Psychology, Fall 2011",
        'video': "",
        'question': """Adaptive physiological changes that occur in response to acute stress include all of the below
EXCEPT
A) suppression of immune system
B) suppression of growth
C) suppression of digestion
D) suppression of cardiovascular tone""",
        'ground_truth': "D) suppression of cardiovascular tone"
    },
    {
        'course': "MIT 9.00SC Introduction to Psychology, Fall 2011",
        'video': "",
        'question': """Which brain region (1) is activated in response to physical pain, the social pain of exclusion,
and the personal pain of romantic rejection, and (2) which brain region shows a relation
between volume and risk for post-traumatic stress disorder (PTSD)?
A) (1) hippocampus; (2) cingulate cortex
B) (1) cingulate cortex; (2) hippocampus
C) (1) amygdala; (2) cingulate cortex
D) (1) cingulate cortex; (2) amygdala""",
        'ground_truth': "B) (1) cingulate cortex; (2) hippocampus"
    },
    {
        'course': "MIT 9.00SC Introduction to Psychology, Fall 2011",
        'video': "",
        'question': """The probability that if one identical (monozygotic) twin is diagnosed with schizophrenia, so will the other twin is
A) 90%
B) 50%
C) 10%
D) 1%""",
        'ground_truth': "B) 50%"
    },
    {
        'course': "MIT 9.00SC Introduction to Psychology, Fall 2011",
        'video': "",
        'question': """The subgenual cingulate has been implicated as playing an important role in depression. Which
one of the results below is FALSE?
A) subgenual cingulate metabolism is reduced in depression
B) subgenual cingulate volume is reduced in depression
C) there is a reduced number of neurons in subgenual cingulate in depression
D) greater activation in the subgenual cingulate predicts better drug treatment outcome""",
        'ground_truth': "C) there is a reduced number of neurons in subgenual cingulate in depression"
    },
    {
        'course': "MIT 9.00SC Introduction to Psychology, Fall 2011",
        'video': "",
        'question': """For ADHD, which is NOT true?
A) treatment with psychostimulants does not slow development of cerebral cortex
B) children with ADHD fail to control response inhibition relative to children without ADHD
C) prefrontal cortical regions in children with ADHD appear to mature structurally two years or more later than in typically developing children
D) activation for reward anticipation is greater in ADHD in the nucleus accumbens than it is in typically developing individuals""",
        'ground_truth': "D) activation for reward anticipation is greater in ADHD in the nucleus accumbens than it is in typically developing individuals"
    },
    {
        'course': "MIT 9.00SC Introduction to Psychology, Fall 2011",
        'video': "",
        'question': """In Milgram's studies of obedience (shocks and learning), which factor did NOT influence the likelihood that subjects would administer shocks to the highest possible level?
A) if experiment was at university or office building
B) if the subject was a man or a woman
C) if the researcher gave no commands once the experiment started
D) if there was an ordinary person (not a scientists) in charge""",
        'ground_truth': "B) if the subject was a man or a woman"
    },
    {
        'course': "MIT 9.00SC Introduction to Psychology, Fall 2011",
        'video': "",
        'question': """What factor has been shown in experiments to increase the willingness for a bystander to help somebody?
A) the presence of passive experimental confederates
B) recent consideration of the importance of helping others
C) being alone
D) the personality factor of extraversion""",
        'ground_truth': "C) being alone"
    },
    {
        'course': "MIT 9.00SC Introduction to Psychology, Fall 2011",
        'video': "",
        'question': """In Festinger's original experiment examining cognitive dissonance, people took a boring test, and were then paid nothing (did not lie) or paid either $1 or $20 to lie and tell the next person that task had been interesting. When later asked to evaluate how truly interesting the task had been, who rated the task as most enjoyable?
A) the people who lied and were paid $20
B) the people who lied and were paid $1
C) the people who did not lie
D) the people who were paid either $1 or $20 equally""",
        'ground_truth': "B) the people who lied and were paid $1"
    },
    {
        'course': "MIT 9.00SC Introduction to Psychology, Fall 2011",
        'video': "",
        'question': """On average, people from a collectivist culture (e.g., Japan) differ from people from an individualistic culture (e.g., United States) in several ways. Which answer below is INCORRECT about ways in which people from collectivist cultures tend to differ from people from individualistic cultures?
A) People from a collectivist culture are more prone to the fundamental attribution error
B) People from a collectivist culture are less susceptible to the attractiveness bias
C) People from a collectivist culture draw more accurately a line that is the same length
relative to a line and a box (frame) previously seen
D) People from a collectivist culture remember an object (like a fish) more accurately when it
is later tested for with the original background""",
        'ground_truth': "A) People from a collectivist culture are more prone to the fundamental attribution error"
    },
    {
        'course': "MIT 9.00SC Introduction to Psychology, Fall 2011",
        'video': "",
        'question': """Experiments have shown people often make errors in predicting what will bring them happiness. All of the following have been shown to NOT make people happier in the long run, except which answer?
A) having many choices
B) a choice that can be changed over the next few days
C) getting tenure
D) expressing gratitude and helping others""",
        'ground_truth': "D) expressing gratitude and helping others"
    },
    {
        'course': "MIT 9.00SC Introduction to Psychology, Fall 2011",
        'video': "",
        'question': """People with Williams syndrome have been tested on the Implicit Association Test (IAT) and exhibited what result?
A) exaggerated gender and racial biases
B) no gender or racial biases
C) a gender bias, but not a racial bias
D) a racial bias, but not a gender bias""",
        'ground_truth': "C) a gender bias, but not a racial bias"
    },
    #
    #
    # --- MIT 8.20 Introduction to Special Relativity, January IAP 2021
    #
    #
    {
        'course': "MIT 8.20 Introduction to Special Relativity, January IAP 2021",
        'video': "",
        'question': """Answer the following question briefly. No calculations are needed. What property of a reference frame makes it an inertial frame?""",
        'ground_truth': "An inertial frame is a reference frame in which Newton’s first law holds: In absence of external forces particles travel in a straight line with a constant velocity."
    },
    {
        'course': "MIT 8.20 Introduction to Special Relativity, January IAP 2021",
        'video': "",
        'question': """Answer the following question briefly. No calculations are needed. Is the following statement true or false: Prior to Einstein, nobody had noticed that the laws of Newtonian mechanics satisfy a principle of relativity""",
        'ground_truth': "False"
    },
    {
        'course': "MIT 8.20 Introduction to Special Relativity, January IAP 2021",
        'video': "",
        'question': """Answer the following question briefly. No calculations are needed. Explain why the Michelson-Morley experiment never yielded any fringe shifts, regardless of the orientation of the experiment, the time of day, or the time of year.""",
        'ground_truth': "Speed of light is the same in all frames. There is no ether."
    },
    {
        'course': "MIT 8.20 Introduction to Special Relativity, January IAP 2021",
        'video': "",
        'question': """Answer the following question briefly. No calculations are needed. Is the following statement true or false: Events A and B occur at the same place in an inertial frame, with A happening before B. It follows that A occurs before B in any other inertial frame.""",
        'ground_truth': "True"
    },
    {
        'course': "MIT 8.20 Introduction to Special Relativity, January IAP 2021",
        'video': "",
        'question': """Answer the following question briefly. No calculations are needed. Critique the statement: The speed of the light emitted by my laser is the same in all frames. So is its frequency""",
        'ground_truth': "Frequency is not an invariant quantity"
    },
    {
        'course': "MIT 8.20 Introduction to Special Relativity, January IAP 2021",
        'video': "",
        'question': """Indicate whether each statement is True or False. No justification is necessary. True / False - A photon can have momentum.""",
        'ground_truth': "True"
    },
    {
        'course': "MIT 8.20 Introduction to Special Relativity, January IAP 2021",
        'video': "",
        'question': """Indicate whether each statement is True or False. No justification is necessary. True / False - In an inertial frame with time coordinate t, the distance between any two objects cannot grow faster than c meters per second.""",
        'ground_truth': "False"
    },
    {
        'course': "MIT 8.20 Introduction to Special Relativity, January IAP 2021",
        'video': "",
        'question': """Indicate whether each statement is True or False. No justification is necessary. True / False - The product of a four-vector with itself is invariant under Lorentz transformation.""",
        'ground_truth': "True"
    },
    {
        'course': "MIT 8.20 Introduction to Special Relativity, January IAP 2021",
        'video': "",
        'question': """Indicate whether each statement is True or False. No justification is necessary. True / False - Consider the two events: “a plane takes oﬀ from Boston” and “six hours later, the plane lands in Los Angeles.” These two events define a timelike interval.""",
        'ground_truth': "True"
    },
    {
        'course': "MIT 8.20 Introduction to Special Relativity, January IAP 2021",
        'video': "",
        'question': """Indicate whether each statement is True or False. No justification is necessary. True / False - For any pair of events A and B, there is always some inertial reference frame in which A and B occur in the same location.""",
        'ground_truth': "False"
    },
    {
        'course': "MIT 8.20 Introduction to Special Relativity, January IAP 2021",
        'video': "",
        'question': """Indicate whether each statement is True or False. No justification is necessary. True / False - The force and acceleration vectors are always parallel.""",
        'ground_truth': "False"
    },
]

test_dataset = [
    #
    #
    # --- CS50's Introduction to Artificial Intelligence with Python 2023
    #
    #
    {
        'course': "CS50's Introduction to Artificial Intelligence with Python 2023",
        'video': "",
        'question': """There are other logical connectives that exist, other than the ones discussed in lecture. One of the most common is “Exclusive Or” (represented using the symbol ⊕). The expression A ⊕ B represents the sentence “A or B, but not both.” Which of the following is logically equivalent to A ⊕ B?
1) (A ∨ B) ∧ ¬ (A ∧ B)
2) (A ∧ B) ∨ ¬ (A ∨ B)
3) (A ∨ B) ∧ (A ∧ B)
4) (A ∨ B) ∧ ¬ (A ∨ B)""",
        'ground_truth': "1) (A ∨ B) ∧ ¬ (A ∧ B)"
    },
    {
        'course': "CS50's Introduction to Artificial Intelligence with Python 2023",
        'video': "",
        'question': """Recall the Bayesian Network shown in lecture, reproduced below.

Rain {none, light, heavy}
  |
  v
Maintenance {yes, no}
  |
  v
Train {on time, delayed}
  |
  v
Appointment {attend, miss}

Rain also directly influences Train:
Rain
  |
  v
Train

Which of the following sentences is true?

1) Assuming we know the train is on time, whether or not there is rain affects the probability that the appointment is attended.
2) Assuming we know there is rain, whether or not there is track maintenance does not affect the probability that the train is on time.
3) Assuming we know there is track maintenance, whether or not there is rain does not affect the probability that the train is on time.
4) Assuming we know the train is on time, whether or not there is track maintenance does not affect the probability that the appointment is attended.
5) Assuming we know there is track maintenance, whether or not there is rain does not affect the probability that the appointment is attended.""",
        'ground_truth': "4) Assuming we know the train is on time, whether or not there is track maintenance does not affect the probability that the appointment is attended."
    },
    {
        'course': "CS50's Introduction to Artificial Intelligence with Python 2023",
        'video': "",
        'question': """Two factories — Factory A and Factory B — design batteries to be used in mobile phones. Factory A produces 60% of all batteries, and Factory B produces the other 40%. 2% of Factory A’s batteries have defects, and 4% of Factory B’s batteries have defects. What is the probability that a battery is both made by Factory A and defective?
Option 1) 0.008
Option 2) 0.012
Option 3) 0.02
Option 4) 0.024
Option 5) 0.028
Option 6) 0.06
Option 7) 0.12
Option 8) 0.2
Option 9) 0.429
Option 10) 0.6
Option 11) None of the above""",
        'ground_truth': "Option 2) 0.012"
    },
    {
        'course': "CS50's Introduction to Artificial Intelligence with Python 2023",
        'video': "",
        'question': """Consider this optimization problem:

A farmer is trying to plant two crops, Crop 1 and Crop 2, and wants to maximize his profits. The farmer will make $500 in profit from each acre of Crop 1 planted, and will make $400 in profit from each acre of Crop 2 planted.

However, the farmer needs to do all of his planting today, during the 12 hours between 7am and 7pm. Planting an acre of Crop 1 takes 3 hours, and planting an acre of Crop 2 takes 2 hours.

The farmer is also limited in terms of supplies: he has enough supplies to plant 10 acres of Crop 1 and enough supplies to plant 4 acres of Crop 2.

Assume the variable C1 represents the number of acres of Crop 1 to plant, and the variable C2 represents the number of acres of Crop 2 to plant.

What are the constraints for this problem?

Option 1) 3 * C1 + 2 * C2 <= 12; C1 <= 10; C2 <= 4
Option 2) 3 * C1 + 2 * C2 <= 12; C1 + C2 <= 14
Option 3) 3 * C1 <= 10; 2 * C2 <= 4
Option 4) C1 + C2 <= 12; C1 + C2 <= 14""",
        'ground_truth': "Option 1) 3 * C1 + 2 * C2 <= 12; C1 <= 10; C2 <= 4"
    },
    {
        'course': "CS50's Introduction to Artificial Intelligence with Python 2023",
        'video': "",
        'question': """Consider a 4x4 grayscale image with the following pixel values.

 2  4  6  8
16 14 12 10
18 20 22 24
32 30 28 26

What would be the result of applying a 2x2 max-pool to the original image?

(Note: Answers are formatted as a matrix [[a, b], [c, d]] where [a, b] is the first row and [c, d] is the second row.)

Option 1) [[16, 12], [32, 28]]
Option 2) [[16, 14], [32, 30]]
Option 3) [[22, 24], [32, 30]]
Option 4) [[14, 12], [30, 28]]
Option 5) [[16, 14], [22, 24]]
Option 6) [[16, 12], [32, 30]]""",
        'ground_truth': "Option 1) [[16, 12], [32, 28]]"
    },
    #
    #
    # --- MIT 9.00SC Introduction to Psychology, Fall 2011
    #
    #
    {
        'course': "MIT 9.00SC Introduction to Psychology, Fall 2011",
        'video': "",
        'question': """In Milgram’s studies of obedience (shocks and learning), which factor did NOT influence the likelihood that subjects would administer shocks to the highest possible level?
    A) if experiment was at university or office building
    B) if the subject was a man or a woman
    C) if the researcher gave no commands once the experiment started
    D) if there was an ordinary person (not a scientists) in charge""",
        'ground_truth': "B) if the subject was a man or a woman"
    },
    {
        'course': "MIT 9.00SC Introduction to Psychology, Fall 2011",
        'video': "",
        'question': """In Festinger’s original experiment examining cognitive dissonance, people took a boring test,
    and were then paid nothing (did not lie) or paid either $1 or $20 to lie and tell the next person
    that task had been interesting. When later asked to evaluate how truly interesting the task had
    been, who rated the task as most enjoyable?
    A) the people who lied and were paid $20
    B) the people who lied and were paid $1
    C) the people who did not lie
    D) the people who were paid either $1 or $20 equally""",
        'ground_truth': "B) the people who lied and were paid $1 "
    },
    {
        'course': "MIT 9.00SC Introduction to Psychology, Fall 2011",
        'video': "",
        'question': """In the Oliver Sacks chapter, the patient with Tourette’s syndrome was treated with Haldol and all EXCEPT which of the below occurred?
    A) he chose eventually to take the medication during the week and not during the weekend
    B) his job situation and home life improved
    C) there was an immediate positive response
    D) he became worse at ping-pong """,
        'ground_truth': "C) there was an immediate positive response"
    },
    {
        'course': "MIT 9.00SC Introduction to Psychology, Fall 2011",
        'video': "",
        'question': """In the Oliver Sacks chapter, all EXCEPT which of the below was true about the woman with “Cupid’s disease”
    A) she had syphilis from her days of prostitution
    B) she became flirtatious at the age of 88
    C) penicillin killed the spirochetes evident in her spinal fluid
    D) penicillin ended her disinhibition""",
        'ground_truth': "D) penicillin ended her disinhibition"
    },
    {
        'course': "MIT 9.00SC Introduction to Psychology, Fall 2011",
        'video': "",
        'question': """Oliver Sacks describes an amnesic patient named "Jimmie G.". The etiology of his amnesia was:
    A) surgery for epilepsy
    B) Alzheimer's disease
    C) Huntington's disease
    D) alcoholism""",
        'ground_truth': "D) alcoholism"
    },
    {
        'course': "MIT 9.00SC Introduction to Psychology, Fall 2011",
        'video': "",
        'question': """Which part of the neuron transmits information away from the cell body toward other neurons?
    Option 1) Axons
    Option 2) Dendrites
    Option 3) Myelin sheath
    Option 4) Soma""",
        'ground_truth': "Option 1) Axons"
    },
    {
        'course': "MIT 9.00SC Introduction to Psychology, Fall 2011",
        'video': "",
        'question': """Which of the following characteristics of experimental design ensure causation when done correctly?
    Option 1) Being sure to have both a dependent and independent variables.
    Option 2) Dividing people into two or more conditions through random assignment.
    Option 3) Doing a statistical analysis of the data.
    Option 4) Making sure that all participants are identical.""",
        'ground_truth': "Option 2) Dividing people into two or more conditions through random assignment."
    },
    {
        'course': "MIT 9.00SC Introduction to Psychology, Fall 2011",
        'video': "",
        'question': """Which of the following types of experimental validity refers to results of a specific experiment that can be generalized to other contexts?
    Option 1) Construct validity
    Option 2) External validity
    Option 3) Internal validity
    Option 4) Researcher validity""",
        'ground_truth': "Option 2) External validity"
    },
    {
        'course': "MIT 9.00SC Introduction to Psychology, Fall 2011",
        'video': "",
        'question': """What type of design prevents biased results due to either the researcher or the participants?
    Option 1) Double blind design
    Option 2) Meta-analysis design
    Option 3) Random assignment design
    Option 4) Replication design""",
        'ground_truth': "Option 1) Double blind design"
    },
    {
        'course': "MIT 9.00SC Introduction to Psychology, Fall 2011",
        'video': "",
        'question': """Each of the three types of cones in the retina is activated primarily by one frequency/color of light. Which of the  following is not actually one of cones?
Option 1) Blue
Option 2) Green
Option 3) Red
Option 4) Yellow""",
        'ground_truth': "Option 4) Yellow"
    },
    #
    #
    # --- MIT 8.20 Introduction to Special Relativity, January IAP 2021
    #
    #
    {
        'course': "MIT 8.20 Introduction to Special Relativity, January IAP 2021",
        'video': "",
        'question': """Indicate whether each statement is True or False. No justification is necessary. True / False - If the density of an object is ρ in its rest mass frame, then when it is observed moving at speed v its density will be ρ' = γ^3 ρ.""",
        'ground_truth': "False"
    },
    {
        'course': "MIT 8.20 Introduction to Special Relativity, January IAP 2021",
        'video': "",
        'question': """A runner is holding a pole that is 12 meters long in its rest frame. He is running towards a barn that is 10 meters long in its rest frame. The barn has two doors: a front door that is initially open, and a back door that is initially closed.
        The runner has speed v = sqrt(3/2) c (γ = 2). He has two friends, one at each door of the barn. The friend at the front door closes the front door as soon as the pole is completely past. The friend at the back door opens the back door just before the pole would hit it.
        
        Yes or No (no justification needed): According to the two friends, are both doors ever closed at the same time?""",
        'ground_truth': "Yes."
    },
    {
        'course': "MIT 8.20 Introduction to Special Relativity, January IAP 2021",
        'video': "",
        'question': """A runner is holding a pole that is 12 meters long in its rest frame. He is running towards a barn that is 10 meters long in its rest frame. The barn has two doors: a front door that is initially open, and a back door that is initially closed.
        The runner has speed v = sqrt(3/2) c (γ = 2). He has two friends, one at each door of the barn. The friend at the front door closes the front door as soon as the pole is completely past. The friend at the back door opens the back door just before the pole would hit it.
        
        Yes or No (no justification needed): According to the runner, are both doors ever closed at the same time?""",
        'ground_truth': "No."
    },
]
