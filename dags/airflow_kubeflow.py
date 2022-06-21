
def batch_train_pred_k8s(state_dict:dict)-> dict:

    from kubernetes.client import V1PodTemplateSpec
    from kubernetes.client import V1ObjectMeta
    from kubernetes.client import V1PodSpec
    from kubernetes.client import V1Container
    from kubernetes.client import V1ResourceRequirements
    from kubernetes.client import CoreV1Api
    from kubeflow.pytorchjob import V1ReplicaSpec
    from kubeflow.pytorchjob import V1PyTorchJob
    from kubeflow.pytorchjob import V1PyTorchJobSpec
    from kubeflow.pytorchjob import PyTorchJobClient
    from kubernetes import config

    config_file = './include/.kube/config'
    
    config.load_kube_config(config_file)

    k8s_client = CoreV1Api()
    if state_dict['k8s_namespace'] not in [item.metadata.name for item in k8s_client.list_namespace().items]:
        k8s_client.create_namespace(client.V1Namespace(metadata=client.V1ObjectMeta(name=state_dict['k8s_namespace'])))

    container = V1Container(
        name="pytorch",
        image=state_dict['train_image'],
        image_pull_policy="Always",
        command=["python", 
                 "/pipeline/load_train.py",
                 "--account="+state_dict['connection_parameters']['account'], 
                 "--password="+state_dict['connection_parameters']['password'],
                 "--username="+state_dict['connection_parameters']['user'],
                 "--role="+state_dict['connection_parameters']['role'], 
                 "--database="+state_dict['connection_parameters']['database'], 
                 "--schema="+state_dict['connection_parameters']['schema'], 
                 "--feature_table_name="+state_dict['feature_table_name'], 
                 "--pred_table_name="+state_dict['pred_table_name']
                ]
    )

    master = V1ReplicaSpec(
        replicas=1,
        restart_policy="OnFailure",
        template=V1PodTemplateSpec(
            spec=V1PodSpec(
                containers=[container]
            )
        )
    )

    worker = V1ReplicaSpec(
        replicas=1,
        restart_policy="OnFailure",
        template=V1PodTemplateSpec(
            spec=V1PodSpec(
                containers=[container]
            )
        )
    )

    pytorchjob = V1PyTorchJob(
        api_version="kubeflow.org/v1",
        kind="PyTorchJob",
        metadata=V1ObjectMeta(name=state_dict['train_job_name'], namespace=state_dict['k8s_namespace']),
        spec=V1PyTorchJobSpec(
            clean_pod_policy="None",
            pytorch_replica_specs={"Master": master} 
        )
    )

    pytorch_client = PyTorchJobClient(config_file=config_file)
    resp = pytorch_client.create(pytorchjob)

    resp = pytorch_client.wait_for_condition(name=resp['metadata']['name'], 
                                           namespace=resp['metadata']['namespace'],
                                           expected_condition='Succeeded')

    state_dict['pytorchjob_uid'] = resp['metadata']['uid']

    return state_dict
