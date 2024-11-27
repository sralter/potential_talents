# Potential Talents - An Apziva Project (#3)
By Samuel Alter  
Apziva: 6bImatZVlK6DnbEo

## Summary<a name='summary'></a>

Using NLP techniques to analyze a job candidate dataset

## Overview<a name='overview'></a>

We are working with a talent sourcing and management company to help them surface candidates that are a best fit for their human resources job post. We are using a dataset of job candidates' job titles, their location, and their number of LinkedIn connections.

### Goals<a name='goals'></a>

Produce a probability, between 0 and 1, of how closely the candidate fits the job description of **"Aspiring human resources"** or **"Seeking human resources."** After an initial recommendation pulls out a candidate(s) to be starred for future consideration, the recommendation will be re-run and new "stars" will be awarded.

To help predict how the candidates fit, we are tracking the performance of two success metrics:
* Rank candidates based on a fitness score
* Re-rank candidates when a candidate is starred

We also need to do the following:
* Explain how the algorithm works and how the ranking improves after each starring iteration
* How to filter out candidates which should not be considered at all
* Determine a cut-off point (if possible) that would work for other roles without losing high-potential candidates
* Ideas to explore on automating this procedure to reduce or eliminate human bias

### The Dataset<a name='dataset'></a>

| Column | Data Type | Comments |
|---|---|---|
| `id` | Numeric | Unique identifier for the candidate |
| `job_title` | Text | Job title for the candidate |
| `location` | Text | Geographic location of the candidate |
| `connections` | Text | Number of LinkedIn connections for the candidate |

Connections over 500 are encoded as "500+". Some do not have specific locations listed and just had their country, so I substituted capitol cities or geographic centers to represent those countries.

## EDA <a name='eda'></a>

There are no nulls in the dataset. There are 104 total observations.

### Connections<a name='connections'></a>

Most applicants have more than 500 connections (n=44). But if we look at [Figure 1](#fig1) to see those that have less than 500 connections, the majority of this group have around 50:

[Figure 1: Histogram of Connections](#fig1)
![Histogram of user connections](figures/3_histogram_connections.jpg)

Viewing the data as a boxplot in Figures [2](#fig2) and [3](#fig3) shows this patten well:

[Figure 2](#fig2): Boxplot of all candidate's connections
![Boxplot of everyone's connections](figures/3_boxplot.jpg)

[Figure 3](#fig3): Boxplot of candidates who have fewer than 500 connections
![Boxplot of those with less than 500 connections](figures/3_boxplot_no500.jpg)

### Geographic Locations<a name='map'></a>

There is some location data and it would be good to see where the candidates are located in the world. [Figure 4](#fig4) shows a choropleth of the candidates' locations.

[Figure 4](#fig4): Choropleth of candidates' locations. Three candidates did not provide a city so they are not included in this map.
![Map of applicants](figures/3_map_choropleth.jpg)

### Initial NLP <a name='nlp-init'></a>

I defined a preprocessor that didn't include lemmatization so as to preserve the specific words included in the job titles. After running the preprocessor over the job titles, I plotted the [10 most-common words](#fig5) in the corpus:

[Figure 5](#fig5): Top 10 most-common words in the candidates' job titles
![Top 10 most common words](figures/3_top10words.jpg)

![Wordcloud](figures/3_wordcloud.jpg)

![Comparing methods](figures/3_methods.jpg)

![Best candidate scores](figures/3_overallscores.jpg)


Under construction...
