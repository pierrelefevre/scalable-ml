import hopsworks

groups = ["winequality", "winequality_balanced", "winequality_typed_balanced"]

for group in groups:

    project = hopsworks.login(project="id2223_pierrelf_emilk2")
    fs = project.get_feature_store()
    fg = fs.get_feature_group(name=group, version=1)
    query = fg.select_all()
    feature_view = fs.get_or_create_feature_view(name=group,
                                                    version=1,
                                                    description="Read from winequality dataset",
                                                    labels=["quality"],
                                                    query=query)
    feature_view.delete_all_training_datasets()