from pipelines.deployement_pipeline import deployement_pipeline ,inference_pipeline

import click

@click.command()
@click.option(
    "--config",
    "-c",
    type = click.Choice(
        [DEPLOY,PREDICT,DEPLOY_AND_PREDICT]
    ),
    default = DEPLOY_AND_PREDICT,
    help = "Optionally you can only choose to run the deployement"
    "pipeline to train and deploy a model (`deploy`), or to"
    "only run a prediction against the deployed model"
    "(`predict`) . By default both will be run "
    "(`deploy_and_predict`)."
)
