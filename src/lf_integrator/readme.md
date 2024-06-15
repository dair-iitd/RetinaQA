# LF Integrator

This section contains commands to run logical form integrator. We have implemented a cache to store query results, so code might be slow initially and will speed up as it progresses.


1. Update the paths in config  : /configs/lf_integrator_config.py
2. Go to src/lf_integrator.
3. Run below command to generate cache file.<br />
    3.1 For grailqa and grailqability<br />
        ```
        python lf_integrator/construct_candidate_logical_forms.py --split $SPLIT --lf_candidates_cache_path data/grailqability/a/lf_integrator/$SPLIT_cache.json --entity_to_type_cache_path data/grailqability/entity_to_type_cache.json --data_type AU &
        ```
        <br />
     3.2 For webqsp<br />
        ```
        python lf_integrator/construct_candidate_logical_forms_webqsp.py --split $SPLIT --lf_candidates_cache_path data/webqsp/lf_integrator/$SPLIT_cache.json --entity_to_type_cache_path data/webqsp/entity_to_type_cache.json
        ```<br />

4. Run the above command separately for all splits - train, dev and test.

Note : 

1. To speed up candidate generation we support generation chunks in parallel, please use the variable start_index and end_index to specify the start and end of chunk to generate from the entire data.
```
python lf_integrator/construct_candidate_logical_forms.py --split train  --start_index 0 --end_index 1000 --lf_candidates_cache_path data/grailqability/a/lf_integrator/train_cache_0_1000.json --entity_to_type_cache_path data/grailqability/entity_to_type_cache.json --data_type A &
```
Merge the candidates of all chunks once generated.

2. It is not necessary to create entity_to_type_cache, it helps to speed up generation.