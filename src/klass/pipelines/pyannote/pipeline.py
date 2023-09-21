"""
This is a boilerplate pipeline 'pyannote'
generated using Kedro 0.18.11
"""
from kedro.pipeline import Pipeline, node
from kedro.pipeline.modular_pipeline import pipeline
from klass.pipelines.pyannote.nodes import concatenate_rttm_files, concatenate_two_text_files, generate_lst_from_rttm_files, generate_uem_from_wav_files

def create_pipeline(**kwargs) -> Pipeline:
    pyannote_preprocess_pipeline = pipeline(
        [
            node(
                    func=concatenate_rttm_files,
                    inputs="rttm_files_input",
                    outputs="rttm_file_output",
                    name="concatenate_rttm_files_node"
            ),
            node(
                    func=generate_lst_from_rttm_files,
                    inputs="rttm_files_input",
                    outputs="lst_file_output",
                    name="generate_lst_from_rttm_files_node"
            ),
            node(
                    func=generate_uem_from_wav_files,
                    inputs="wav_files_duration_input",
                    outputs="uem_file_output",
                    name="generate_uem_from_wav_files_node"
            ),        
        ]
    )

    merge_ali_ami_pipeline = pipeline(
        [
            node(
                func=concatenate_two_text_files,
                inputs=["ali_far_rttm_file_output", "ami_far_rttm_file_output"],
                outputs="combined_rttm_file_output",
                name="concatenate_two_rttm_files_node"
            ),
            node(
                func=concatenate_two_text_files,
                inputs=["ali_far_uem_file_output", "ami_far_uem_file_output"],
                outputs="combined_uem_file_output",
                name="concatenate_two_uem_files_node"
            ),
            node(
                func=concatenate_two_text_files,
                inputs=["ali_far_lst_file_output", "ami_far_lst_file_output"],
                outputs="combined_lst_file_output",
                name="concatenate_two_lst_files_node"
            )                        
        ]
    )

    ali_far_val_pipeline = pipeline(
        pipe=pyannote_preprocess_pipeline,
        namespace="ali_far_val"
    )

    ami_far_val_pipeline = pipeline(
        pipe=pyannote_preprocess_pipeline,
        namespace="ami_far_val"
    )

    ali_far_test_pipeline = pipeline(
        pipe=pyannote_preprocess_pipeline,
        namespace="ali_far_test"
    )

    ami_far_test_pipeline = pipeline(
        pipe=pyannote_preprocess_pipeline,
        namespace="ami_far_test"
    )    

    ali_far_train_pipeline = pipeline(
        pipe=pyannote_preprocess_pipeline,
        namespace="ali_far_train"
    )    

    ami_far_train_pipeline = pipeline(
        pipe=pyannote_preprocess_pipeline,
        namespace="ami_far_train"
    )

    combined_train_pipeline = pipeline(
        pipe=merge_ali_ami_pipeline,
        inputs={
            "ali_far_rttm_file_output": "ali_far_train.rttm_file_output",
            "ami_far_rttm_file_output": "ami_far_train.rttm_file_output",
            "ali_far_uem_file_output": "ali_far_train.uem_file_output",
            "ami_far_uem_file_output": "ami_far_train.uem_file_output",
            "ali_far_lst_file_output": "ali_far_train.lst_file_output",
            "ami_far_lst_file_output": "ami_far_train.lst_file_output"
            },
        outputs={
            "combined_rttm_file_output": "combined_train_rttm_file_output",
            "combined_uem_file_output": "combined_train_uem_file_output",
            "combined_lst_file_output": "combined_train_lst_file_output"
        },
        namespace="combined_train"
    )

    combined_val_pipeline = pipeline(
        pipe=merge_ali_ami_pipeline,
        inputs={
            "ali_far_rttm_file_output": "ali_far_val.rttm_file_output",
            "ami_far_rttm_file_output": "ami_far_val.rttm_file_output",
            "ali_far_uem_file_output": "ali_far_val.uem_file_output",
            "ami_far_uem_file_output": "ami_far_val.uem_file_output",
            "ali_far_lst_file_output": "ali_far_val.lst_file_output",
            "ami_far_lst_file_output": "ami_far_val.lst_file_output"
            },
        outputs={
            "combined_rttm_file_output": "combined_val_rttm_file_output",
            "combined_uem_file_output": "combined_val_uem_file_output",
            "combined_lst_file_output": "combined_val_lst_file_output"
        },
        namespace="combined_val"
    )

    combined_test_pipeline = pipeline(
        pipe=merge_ali_ami_pipeline,
        inputs={
            "ali_far_rttm_file_output": "ali_far_test.rttm_file_output",
            "ami_far_rttm_file_output": "ami_far_test.rttm_file_output",
            "ali_far_uem_file_output": "ali_far_test.uem_file_output",
            "ami_far_uem_file_output": "ami_far_test.uem_file_output",
            "ali_far_lst_file_output": "ali_far_test.lst_file_output",
            "ami_far_lst_file_output": "ami_far_test.lst_file_output"
            },
        outputs={
            "combined_rttm_file_output": "combined_test_rttm_file_output",
            "combined_uem_file_output": "combined_test_uem_file_output",
            "combined_lst_file_output": "combined_test_lst_file_output"
        },
        namespace="combined_test"        
    )

    return ali_far_val_pipeline + ami_far_val_pipeline + ali_far_test_pipeline + ami_far_test_pipeline + ali_far_train_pipeline + ami_far_train_pipeline + combined_train_pipeline + combined_val_pipeline + combined_test_pipeline
