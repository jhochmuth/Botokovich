# Botokovich
A project to investigate the applicability of deep learning to music generation.
Please note: this project is still in an early experimental stage.

Example of generated music in staff notation:

![Generated music img](https://github.com/jhochmuth/Botokovich/blob/master/data/generated_examples/exemplary_examples/chorales/sheetmusic_300hs_10bs_001lr_40e_0.png)

Visit this [link](https://drive.google.com/file/d/1MtvIiXnByA16t_IBB8ZZC_4ppqlmgbHT/view?usp=sharing) to hear the midi of this example. This example was created using a model trained on Bach chorales. I am unable to get Quicktime player to play the generated midi file. If it doesn't work for you, them try playing the file with [this site](https://onlinesequencer.net/import).

## Specifics
Music notation contains a large number of similarities to language.
It is probable that neural networks used for NLP tasks can be applied to music. In fact, many successful results have already been documented in a number of papers.
This project will investigate different methods of generating music and evaluate their success.
If the project leads to generated music that is successful enough, a web application will be created making use of the trained models.

## Related Research
TODO: Document academic papers investigating music generation.

## Examples of Generated Pieces
Examples of generated music can be found in the "data/generated_examples" directory. Examples created using a model trained on only Bach chorales will be found in the "chorales" directory, while those created using a model trained on baroque and classical era pieces (excluding Bach chorales) are found in the "major" directory (the name of the directory alludes to the fact that separate models are trained for major and minor key pieces).

So far, all generated examples have been created by feeding the same fragment to the trained language model. Some of the hyperparameter settings used to generate a specific sequence can be found in the name of each file.

Only examples that were judged to have at least some musical value were kept.

A commendable sequence is a sequence which contains at least one short phrase that is musically interesting. It is acceptable for a majority of the sequence to be musically boring or even include "mistakes" as long as it fulfills this qualification.

An exemplary sequence is a sequence which contains only some "mistakes" and is also musically interesting for a significant portion of the sequence.

Because of the nature of music, there is necessarily a large amount of ambiguity in judging the musical value of a sequence. 
