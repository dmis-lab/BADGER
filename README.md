# BADGER: Biologically-Aware Interpretable Differential Gene Expression Ranking Model

## Abstract
Understanding which genes are significantly influenced by a drug can reveal insights into the mechanism of action, a vital aspect of drug repurposing. A drug affecting specific pathways or gene expressions in one disease could potentially be effective in another with similar genetic patterns. Ranking genes according to the extent of their expression change within cells, pre- and post-drug treatment, allows us to identify the genes most substantially affected by a particular drug. However, previous studies' limited scope of cells and explainability constraints hinder a comprehensive understanding of drug-cell response. We introduce BADGER, a Biologically-Aware interpretable Differential Gene Expression Ranking model. This model is designed to predict gene expression changes resulting from interactions between cancer cell lines and chemical compounds. It employs a similarity-based method for representing novel and diverse cancer cell lines. Additionally, the three attention blocks in the model mimic the cascading effects of chemical compounds, ensuring a thorough consideration of their complex interactions with cancer cell lines. Moreover, the integration of prior knowledge about drugs' target into the model enhances its explainability. Comparative evaluations demonstrate that BADGER outperforms baseline models in capturing the intricate interaction between cancer cell lines and chemical compounds. Its application in drug repurposing is further validated by analyzing the attention maps and predicted rankings of differentially expressed genes for both approved and repurposing candidate drugs, highlighting its potential in identifying novel therapeutic uses for existing drugs. (*submitted to Bioinformatics, under review*)

## Dataset

You can download the preprocessed dataset from this [google drive link](https://drive.google.com/drive/folders/19-qR-TDAKOAc00_nSU4IIu63IWFnZ4YJ?usp=sharing).

## Contributors

<table>
	<tr>
		<th>Name</th>		
		<th>Affiliation</th>
		<th>Email</th>
	</tr>
	<tr>
		<td>Hajung Kim&dagger;</td>		
		<td>Data Mining and Information Systems Lab,<br>Korea University, Seoul, South Korea</td>
		<td>hajungk@korea.ac.kr</td>
	</tr>
	<tr>
		<td>Mogan Gim&dagger;</td>		
		<td>Data Mining and Information Systems Lab,<br>Korea University, Seoul, South Korea</td>
		<td>akim@korea.ac.kr</td>
	</tr>
	<tr>
		<td>Seungheun Baek</td>		
		<td>Data Mining and Information Systems Lab,<br>Korea University, Seoul, South Korea</td>
		<td>sheunbaek@korea.ac.kr</td>
	</tr>
	<tr>
		<td>Soyon Park</td>		
		<td>Data Mining and Information Systems Lab,<br>Korea University, Seoul, South Korea</td>
		<td>soyon0304@gmail.com</td>
	</tr>
	<tr>
		<td>Sunkyu Kim*</td>		
		<td>AIGEN Sciences, Seoul, South Korea</td>
		<td>sunkyu-kim@korea.ac.kr</td>
	</tr>
	<tr>
		<td>Jaewoo Kang*</td>		
		<td>Data Mining and Information Systems Lab,<br>Korea University, Seoul, South Korea</td>
		<td>kangj@korea.ac.kr</td>
	</tr>

</table>

- &dagger;: *Equal Contributors.*
- &ast;: *Corresponding Author*