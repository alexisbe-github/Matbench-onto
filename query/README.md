# SPARQL queries for the performance figures

This directory documents the SPARQL queries used to extract the data for the
model-performance figures. All performance values correspond to the MAE on the
Matbench Discovery convex-hull-distance regression task and the
`full_test_set` split.

## Performance as a function of parameter count

This query returns the model name, parameter count, MAE, and explicit
architecture type.

```sparql
PREFIX arch:    <https://k.loria.fr/ontologies/architectureonto#>
PREFIX eval:    <https://k.loria.fr/ontologies/evaluationonto#>
PREFIX evalind: <https://k.loria.fr/ontologies/evaluationonto-individuals#>
PREFIX rdfs:    <http://www.w3.org/2000/01/rdf-schema#>

SELECT DISTINCT
       ?modelName
       ?parameterCount
       ?performance
       ?architectureType
       ?architectureTypeName
WHERE {
    ?model
        arch:hasParameterNumber ?parameterCount ;
        arch:hasMachineLearningArchitecture ?architecture .

    ?architecture a ?architectureType .

    VALUES ?architectureType {
        arch:GraphConvolutionArchitecture
        arch:MessagePassingNeuralNetworkArchitecture
        arch:EquivariantMessagePassingNeuralNetworkArchitecture
        arch:GraphNeuralNetworkArchitecture
        arch:GraphTransformerArchitecture
        arch:TransformerArchitecture
        arch:AttentionBasedArchitecture
    }

    ?evaluation
        eval:evaluatesModelVariant ?model ;
        eval:producesBenchmarkResult ?result .

    ?result
        eval:hasResultTask
            evalind:convex_hull_distance_regression_task ;
        eval:hasMetricResult ?metricResult .

    ?metricResult
        eval:hasMetricType evalind:mae_metric_type ;
        eval:hasMetricValue ?performance .

    FILTER(CONTAINS(STR(?metricResult), "full_test_set"))

    OPTIONAL {
        ?model rdfs:label ?modelLabel .
        FILTER(
            LANG(?modelLabel) = "" ||
            LANGMATCHES(LANG(?modelLabel), "en")
        )
    }

    BIND(
        COALESCE(
            STR(?modelLabel),
            REPLACE(STR(?model), "^.*[#/]", "")
        )
        AS ?modelName
    )

    BIND(
        REPLACE(STR(?architectureType), "^.*[#/]", "")
        AS ?architectureTypeName
    )
}
ORDER BY ?architectureTypeName ?parameterCount
```

## Performance as a function of Matbench Discovery release date

This query replaces the parameter count with the date on which the model was
added to Matbench Discovery. The date is stored as an `xsd:date`.

```sparql
PREFIX arch:    <https://k.loria.fr/ontologies/architectureonto#>
PREFIX eval:    <https://k.loria.fr/ontologies/evaluationonto#>
PREFIX evalind: <https://k.loria.fr/ontologies/evaluationonto-individuals#>
PREFIX rdfs:    <http://www.w3.org/2000/01/rdf-schema#>

SELECT DISTINCT
       ?modelName
       ?releaseDate
       ?performance
       ?architectureType
       ?architectureTypeName
WHERE {
    ?model
        arch:hasReleaseDate ?releaseDate ;
        arch:hasMachineLearningArchitecture ?architecture .

    ?architecture a ?architectureType .

    VALUES ?architectureType {
        arch:GraphConvolutionArchitecture
        arch:MessagePassingNeuralNetworkArchitecture
        arch:EquivariantMessagePassingNeuralNetworkArchitecture
        arch:GraphNeuralNetworkArchitecture
        arch:GraphTransformerArchitecture
        arch:TransformerArchitecture
        arch:AttentionBasedArchitecture
    }

    ?evaluation
        eval:evaluatesModelVariant ?model ;
        eval:producesBenchmarkResult ?result .

    ?result
        eval:hasResultTask
            evalind:convex_hull_distance_regression_task ;
        eval:hasMetricResult ?metricResult .

    ?metricResult
        eval:hasMetricType evalind:mae_metric_type ;
        eval:hasMetricValue ?performance .

    FILTER(CONTAINS(STR(?metricResult), "full_test_set"))

    OPTIONAL {
        ?model rdfs:label ?modelLabel .
        FILTER(
            LANG(?modelLabel) = "" ||
            LANGMATCHES(LANG(?modelLabel), "en")
        )
    }

    BIND(
        COALESCE(
            STR(?modelLabel),
            REPLACE(STR(?model), "^.*[#/]", "")
        )
        AS ?modelName
    )

    BIND(
        REPLACE(STR(?architectureType), "^.*[#/]", "")
        AS ?architectureTypeName
    )
}
ORDER BY ?releaseDate ?architectureTypeName ?performance
```

## Five most frequently used training datasets

This query counts distinct training runs and model variants for each training
dataset. `MP 2022` and `Materials Project` are grouped into a single
`Materials Project` category before ranking.

```sparql
PREFIX train:   <https://k.loria.fr/ontologies/trainingonto#>
PREFIX dataind: <https://k.loria.fr/ontologies/dataset-individuals#>
PREFIX rdfs:    <http://www.w3.org/2000/01/rdf-schema#>

SELECT
       ?datasetName
       (COUNT(DISTINCT ?trainingRun) AS ?numberOfTrainingRuns)
       (COUNT(DISTINCT ?model) AS ?numberOfModels)
WHERE {
    ?trainingRun
        train:usesDataset ?dataset ;
        train:trainsModelVariant ?model .

    OPTIONAL {
        ?dataset rdfs:label ?originalDatasetName .
    }

    BIND(
        IF(
            ?dataset IN (
                dataind:mp_2022,
                dataind:materials_project
            ),
            dataind:materials_project,
            ?dataset
        )
        AS ?groupedDataset
    )

    BIND(
        IF(
            ?groupedDataset = dataind:materials_project,
            "Materials Project",
            COALESCE(
                STR(?originalDatasetName),
                REPLACE(STR(?dataset), "^.*[#/]", "")
            )
        )
        AS ?datasetName
    )
}
GROUP BY ?groupedDataset ?datasetName
ORDER BY DESC(?numberOfModels)
```

## Notes

- Architecture types are restricted to explicit RDF types listed in each
  query; disable inference to the query to avoid wrong results.

