# BADGER

### Abstract
**Motivation** 
Understanding which genes are significantly influenced by a drug provides insights into its mechanism of action, crucial for drug repurposing. A drug that targets specific pathways or gene expressions in one disease might also be effective in another with similar genetic profiles. By ranking genes according to the extent of their expression changes in cells before and after drug treatment, we can identify the genes most impacted by the drug. However, the limited range of cell lines in previous studies and constraints on explainability have hindered comprehensive understanding of drug-cell responses.
**Result** 
We introduce BADGER, a Biologically-Aware interpretable Differential Gene Expression Ranking model. BADGER is a robust and interpretable model designed to predict gene expression changes resulting from interactions between cancer cell lines and chemical compounds. BADGER effectively handles explainability by integrating prior knowledge of drug targets through pathway information, and addresses novel cancer cell lines through a similarity-based embedding method. It employs three attention blocks that mimic the cascading effects of chemical compounds, ensuring a comprehensive understanding of their complex interactions with cancer cell lines. BADGER's generalization capabilities are rigorously validated: it demonstrates superior performance over baseline models in unseen cell and unseen pair split evaluations, showcasing its ability to robustly predict gene expression changes for untested drug-cell line combinations. Based on these results, BADGER exhibits its potential in drug repurposing scenarios, particularly in providing therapeutic plans for new or resistant diseases by leveraging similarities with other diseases.

---
### Running Models

#### Training
```bash
python run.py -sn {session_name} -sf {start_fold_num} -ef {end_fold_num}
```

#### Testing
```bash
python run.py -sn {session_name} -sf {start_fold_num} -ef {end_fold_num} --test
```

#### Debug Mode
```bash
python run.py -sn {session_name} -sf {start_fold_num} -ef {end_fold_num} --debug_mode
```

### Available Session Names
- badger: Our proposed model
- MLP, DeepCE, CIGER: Baseline models
- badger_light: Ablation study model (without perturbation-pathway cross attention)


### Example
```bash
# Train badger model
python run.py -sn badger -sf 1 -ef 1

# Test CIGER model
python run.py -sn ciger -sf 1 -ef 1 --test

# Debug mode with MLP
python run.py -sn mlp -sf 1 -ef 1 --debug_mode
```


### Cell Line Similarity Process
```bash
#The cell line similarity calculation process can be found in the Jupyter notebook:
Copy./src/calculate_cell_embeddings.ipynb
```

### Data Availability
All related datasets are available through our Google Drive link: https://drive.google.com/file/d/11H4ZZJkteb9L5y6JJxdAE2F6k4qQjNG2/view?usp=drive_link




### Contributors

<table>
	<tr>
		<th>Name</th>		
		<th>Affiliation</th>
		<th>Email</th>
	</tr>
	<tr>
		<td>Hajung Kim&dagger;</td>		
		<td>Department of Computer Science,<br>Korea University, Seoul, South Korea</td>
		<td>hajungk@korea.ac.kr</td>
	</tr>	
	<tr>
		<td>Mogan Gim&dagger;</td>		
		<td>Department of Biomedical Engineering,<br>Hankuk University of Foreign Studies, Yongin, South Korea</td>
		<td>gimmogan@hufs.ac.kr</td>
	</tr>
	<tr>
		<td>Seungheun Baek</td>		
		<td>Department of Computer Science,<br>Korea University, Seoul, South Korea</td>
		<td>sheunbaek@korea.ac.kr</td>
	</tr>
		<tr>
		<td>Soyon Park</td>		
		<td>Department of Computer Science,<br>Korea University, Seoul, South Korea</td>
		<td>soyon_park@korea.ac.kr</td>
	</tr>
	<tr>
		<td>Jaewoo Kang*</td>		
		<td>Department of Computer Science,<br>Korea University, Seoul, South Korea</td>
		<td>kangj@korea.ac.kr</td>
	</tr>
</table>

- &dagger;: *Equal Contributors*
- &ast;: *Corresponding Author*
