Download Link: https://assignmentchef.com/product/solved-csye7245-assignment4
<br>
The goal of the first part of this assignment is to create API that

<ol>

 <li>Anonmyizes the data through:</li>

</ol>

<ul>

 <li>Masking</li>

 <li>Anonymization</li>

</ul>

Then, building upon the Infrastructure for login and server less functions using Cognito in

Assignment 1, integrate the APIs so that

<ol>

 <li>Only authenticated users can call these APIs</li>

 <li>Use Amazon Step functions and Lamda functions to make it server less where feasible (This is a design decision; You may host servers and then call those APIs or call readily available APIs like Amazon Comprehend through lambda functions) Refer:</li>

 <li>Complete and submit the following tutorials:</li>

</ol>

<a href="https://aws.amazon.com/blogs/machine-learning/detecting-and-redacting-pii-using-amazon-comprehend/">https://aws.amazon.com/blogs/machine-learning/detecting-and-redacting-pii-using-amazon-co </a><a href="https://aws.amazon.com/blogs/machine-learning/detecting-and-redacting-pii-using-amazon-comprehend/">mprehend/</a>

<a href="https://aws.amazon.com/blogs/machine-learning/detecting-and-redacting-pii-using-amazon-comprehend/">(Links to an external site.)</a>

<ol start="2">

 <li>Presidio: <a href="https://github.com/microsoft/presidio">https://github.com/microsoft/presidio</a></li>

</ol>

1

<strong>Implementation:</strong>

——————-

<strong><u>Create three APIs:</u></strong>

<strong><u>API 1: Access</u></strong>

This API should retrieve the EDGAR filings data from the S3 bucket

<strong><u>API 2: Named entity recognition</u></strong>

This API should take a link to a file on S3 and:

<ul>

 <li>Call Amazon Comprehend OR Google OR Presidio or any tool of your choice to find entities. (You can define the list of entities or use the default ones like Name, SSN, Date etc.)</li>

 <li>Store these on S3</li>

</ul>

<strong><u>API 3: Implement masking, and anonymization functions.</u></strong>

Note: You have to define the API so as to indicate which entities need to be masked, which needs to be anonymized. You also need to get the location of the file/files as input and output the files back to S3. You can choose a method of your choice!

<strong>Part 2:</strong>

In this part of the assignment, you will build upon the pre-processed (anomymized/masked data) and build a sentiment analysis model that could take the location of the anonymized file as an input and generate sentiments for each sentence.

To build this service, you need a Sentiment analysis model that has been trained on “labeled”, “Edgar” datasets. Note that you need to have labeled training data which means someone has to label the statements. We will use the IMDB dataset as a proxy and build a sentiment analyzer that can be tested on the anonymized datasets you prepared in the prior assignment

<strong>Preparation:</strong>

WHy TFX?

Read <a href="https://blog.tensorflow.org/2020/09/brief-history-of-tensorflow-extended-tfx.html#TFX">https://blog.tensorflow.org/2020/09/brief-history-of-tensorflow-extended-tfx.html#TFX</a> for the history of TFX and MLOps Watch for an overview:

<a href="https://www.youtube.com/watch?v=YeuvR6m6ACQ&amp;list=PLQY2H8rRoyvxR15n04JiW0ezF5HQRs_8F&amp;index=1">https://www.youtube.com/watch?v=YeuvR6m6ACQ&amp;list=PLQY2H8rRoyvxR15n04JiW0ezF5HQRs_8F </a><a href="https://www.youtube.com/watch?v=YeuvR6m6ACQ&amp;list=PLQY2H8rRoyvxR15n04JiW0ezF5HQRs_8F&amp;index=1">&amp;index=1</a>

<strong>Goal: </strong><strong>To deploy a sentiment analysis model to create a Model-as-a-service for anonymized data</strong>

<h1>Step 1:Train TensorFlow models using TensorFlow Extended (TFX)</h1>

Replicate the architecture to train the model for the anonymized data using BERT and this architecture that leverages <strong>TensorFlow Hub</strong>, <strong>Tensorflow Transform, </strong><strong>TensorFlow Data</strong>

<strong>Validation and Tensorflow Text and Tensorflow Serving</strong>

<strong>The pipeline takes advantage of the broad TensorFlow Ecosystem, including:</strong>

<ul>

 <li><strong>Loading the IMDB dataset via TensorFlow Datasets</strong></li>

 <li><strong>Loading a pre-trained model via tf.hub</strong></li>

 <li><strong>Manipulating the raw input data with tf.text</strong></li>

 <li><strong>Building a simple model architecture with Keras</strong></li>

 <li><strong>Composing the model pipeline with TensorFlow Extended, e.g. TensorFlow Transform, TensorFlow Data Validation and then consuming the tf.Keras model with the latest Trainer component from TFX</strong></li>

</ul>

<strong>Ref:</strong>

<a href="https://blog.tensorflow.org/2020/03/part-1-fast-scalable-and-accurate-nlp-tensorflow-deploying-bert.html">https://blog.tensorflow.org/2020/03/part-1-fast-scalable-and-accurate-nlp-tensorflow-deploying-b </a><a href="https://blog.tensorflow.org/2020/03/part-1-fast-scalable-and-accurate-nlp-tensorflow-deploying-bert.html">ert.html </a><a href="https://blog.tensorflow.org/2020/06/part-2-fast-scalable-and-accurate-nlp.html">https://blog.tensorflow.org/2020/06/part-2-fast-scalable-and-accurate-nlp.html </a>Sample Code:

<a href="https://colab.research.google.com/github/tensorflow/workshops/blob/master/blog/TFX_Pipeline_for_Bert_Preprocessing.ipynb#scrollTo=WWni3fVVafDa">https://colab.research.google.com/github/tensorflow/workshops/blob/master/blog/TFX_Pipeline_f </a><a href="https://colab.research.google.com/github/tensorflow/workshops/blob/master/blog/TFX_Pipeline_for_Bert_Preprocessing.ipynb#scrollTo=WWni3fVVafDa">or_Bert_Preprocessing.ipynb#scrollTo=WWni3fVVafDa</a>

Note: Use AlBert instead of BERT (<a href="https://tfhub.dev/google/albert_base/3">https://tfhub.dev/google/albert_base/3</a>)

Also, Note, you will be implementing sentiment analysis.. So you will have to change the dataset.

Use the <a href="https://www.tensorflow.org/api_docs/python/tf/keras/datasets/imdb">IMDB dataset</a> as a a proxy. See

<a href="https://www.tensorflow.org/tutorials/keras/text_classification_with_hub">(</a><a href="https://www.tensorflow.org/tutorials/keras/text_classification_with_hub">https://www.tensorflow.org/tutorials/keras/text_classification_with_hub</a> ) for details.

<h1>Step 2: Serve the model as a REST API</h1>

<strong>Use the TENSORFLOW TFX RESTFUL API to serve the model </strong><a href="https://www.tensorflow.org/tfx/serving/api_rest">(</a><a href="https://www.tensorflow.org/tfx/serving/api_rest">https://www.tensorflow.org/tfx/serving/api_rest</a> ) See for sample code:

<a href="https://www.tensorflow.org/tfx/tutorials/serving/rest_simple">https://www.tensorflow.org/tfx/tutorials/serving/rest_simple</a>

OR

<a href="https://towardsdatascience.com/serving-image-based-deep-learning-models-with-tensorflow-servings-restful-api-d365c16a7dc4">https://towardsdatascience.com/serving-image-based-deep-learning-models-with-tensorflow-servi </a><a href="https://towardsdatascience.com/serving-image-based-deep-learning-models-with-tensorflow-servings-restful-api-d365c16a7dc4">ngs-restful-api-d365c16a7dc4</a>

<strong>OR</strong>

<strong>USE FAST API to serve the model </strong>See for sample code:

<a href="https://medium.com/python-data/how-to-deploy-tensorflow-2-0-models-as-an-api-service-with-fastapi-docker-128b177e81f3">https://medium.com/python-data/how-to-deploy-tensorflow-2-0-models-as-an-api-service-with-fas </a><a href="https://medium.com/python-data/how-to-deploy-tensorflow-2-0-models-as-an-api-service-with-fastapi-docker-128b177e81f3">tapi-docker-128b177e81f3</a>

OR

<a href="https://testdriven.io/blog/fastapi-streamlit/">https://testdriven.io/blog/fastapi-streamlit/</a>

<strong>Step 3:Dockerize the API service.</strong>

For sample code on how to Dockerize API:

See <a href="https://www.tensorflow.org/tfx/serving/docker">https://www.tensorflow.org/tfx/serving/docker</a>

OR <a href="https://medium.com/python-data/how-to-deploy-tensorflow-2-0-models-as-an-api-service-with-fastapi-docker-128b177e81f3">https://medium.com/python-data/how-to-deploy-tensorflow-2-0-models-as-an-api-service-with-fas </a><a href="https://medium.com/python-data/how-to-deploy-tensorflow-2-0-models-as-an-api-service-with-fastapi-docker-128b177e81f3">tapi-docker-128b177e81f3</a>

<h1>Step 4: Build a Reference App in Streamlit to test the API</h1>

Input: Link to an anonymized/deanonymized file in Amazon S3 Output: Sentiment scores.

Sample Code:

See <a href="https://testdriven.io/blog/fastapi-streamlit/">https://testdriven.io/blog/fastapi-streamlit/</a> for samples

<h1>Step 5: Write unit tests &amp; Load tests to test your api</h1>

<ul>

 <li>You will have to show test cases you have tested (using pytest)</li>

 <li>Load test the API with Locust</li>

</ul>