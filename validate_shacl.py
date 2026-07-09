import argparse
import sys
from pathlib import Path

from pyshacl import validate
from rdflib import Graph, Namespace, RDF, OWL, RDFS, URIRef


BASE_DIR = Path(__file__).resolve().parent

DEFAULT_TTL_DIR = BASE_DIR / "outputs" / "ttl"
ONTOLOGY_FILES = [
    BASE_DIR / "ontology" / "architecture.ttl",
    BASE_DIR / "ontology" / "trainingonto.ttl",
    BASE_DIR / "ontology" / "datasetonto.ttl",
    BASE_DIR / "ontology" / "evaluationonto.ttl",
]

ARCH = Namespace("https://k.loria.fr/ontologies/architectureonto#")
TRAIN = Namespace("https://k.loria.fr/ontologies/trainingonto#")
DATA = Namespace("https://k.loria.fr/ontologies/datasetonto#")
EVAL = Namespace("https://k.loria.fr/ontologies/evaluationonto#")

ONTOLOGY_NAMESPACES = (
    str(ARCH),
    str(TRAIN),
    str(DATA),
    str(EVAL),
)

SHACL_SHAPES = """
@prefix sh: <http://www.w3.org/ns/shacl#> .
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .

@prefix arch: <https://k.loria.fr/ontologies/architectureonto#> .
@prefix train: <https://k.loria.fr/ontologies/trainingonto#> .
@prefix data: <https://k.loria.fr/ontologies/datasetonto#> .
@prefix eval: <https://k.loria.fr/ontologies/evaluationonto#> .



arch:MachineLearningArchitectureClassShape
    a sh:NodeShape ;
    sh:or (
        [ sh:class arch:MachineLearningArchitecture ]
        [ sh:class arch:NeuralNetworkArchitecture ]
        [ sh:class arch:AttentionBasedArchitecture ]
        [ sh:class arch:EquivariantGraphNeuralNetworkArchitecture ]
        [ sh:class arch:EquivariantMessagePassingNeuralNetworkArchitecture ]
        [ sh:class arch:EquivariantNeuralNetworkArchitecture ]
        [ sh:class arch:GatedNeuralNetworkArchitecture ]
        [ sh:class arch:GraphConvolutionArchitecture ]
        [ sh:class arch:GraphNeuralNetworkArchitecture ]
        [ sh:class arch:GraphTransformerArchitecture ]
        [ sh:class arch:MessagePassingNeuralNetworkArchitecture ]
        [ sh:class arch:TransformerArchitecture ]
    ) .

arch:ModelVariantClassShape
    a sh:NodeShape ;
    sh:or (
        [ sh:class arch:ModelVariant ]
        [ sh:class train:ModelMaterialization ]
        [ sh:class train:Checkpoint ]
    ) .

train:ModelMaterializationClassShape
    a sh:NodeShape ;
    sh:or (
        [ sh:class train:ModelMaterialization ]
        [ sh:class train:Checkpoint ]
    ) .

train:TrainingRunClassShape
    a sh:NodeShape ;
    sh:or (
        [ sh:class train:MachineLearningTrainingRun ]
        [ sh:class train:MachineLearningActiveLearningRun ]
        [ sh:class train:MachineLearningAdaptationRun ]
        [ sh:class train:MachineLearningDistillationRun ]
        [ sh:class train:MachineLearningFinetuningRun ]
        [ sh:class train:MachineLearningLoRAFineTuningRun ]
        [ sh:class train:MachineLearningTransferLearningRun ]
        [ sh:class train:PretrainingRun ]
    ) .

train:ObjectiveFunctionClassShape
    a sh:NodeShape ;
    sh:or (
        [ sh:class train:ObjectiveFunction ]
        [ sh:class train:LossFunction ]
    ) .

data:DatasetClassShape
    a sh:NodeShape ;
    sh:or (
        [ sh:class data:Dataset ]
        [ sh:class data:DatasetSplit ]
        [ sh:class data:LabeledDataset ]
        [ sh:class train:TrainingDataset ]
    ) .

data:TrainingDatasetClassShape
    a sh:NodeShape ;
    sh:or (
        [ sh:class train:TrainingDataset ]
        [ sh:class data:LabeledDataset ]
    ) .

eval:TaskClassShape
    a sh:NodeShape ;
    sh:or (
        [ sh:class eval:Task ]
        [ sh:class eval:ClassificationTask ]
        [ sh:class eval:RegressionTask ]
    ) .


eval:EvaluatesModelVariantSubjectShape
    a sh:NodeShape ;
    sh:targetSubjectsOf eval:evaluatesModelVariant ;
    sh:class eval:MachineLearningEvaluationRun ;
    sh:message "eval:evaluatesModelVariant must have an eval:MachineLearningEvaluationRun as subject." .

eval:EvaluatesModelVariantObjectShape
    a sh:NodeShape ;
    sh:targetObjectsOf eval:evaluatesModelVariant ;
    sh:node arch:ModelVariantClassShape ;
    sh:message "eval:evaluatesModelVariant must point to an arch:ModelVariant." .

eval:EvaluatesTaskSubjectShape
    a sh:NodeShape ;
    sh:targetSubjectsOf eval:evaluatesTask ;
    sh:class eval:MachineLearningEvaluationRun ;
    sh:message "eval:evaluatesTask must have an eval:MachineLearningEvaluationRun as subject." .

eval:EvaluatesTaskObjectShape
    a sh:NodeShape ;
    sh:targetObjectsOf eval:evaluatesTask ;
    sh:node eval:TaskClassShape ;
    sh:message "eval:evaluatesTask must point to an eval:Task." .

eval:UsesBenchmarkReleaseSubjectShape
    a sh:NodeShape ;
    sh:targetSubjectsOf eval:usesBenchmarkRelease ;
    sh:class eval:MachineLearningEvaluationRun ;
    sh:message "eval:usesBenchmarkRelease must have an eval:MachineLearningEvaluationRun as subject." .

eval:UsesBenchmarkReleaseObjectShape
    a sh:NodeShape ;
    sh:targetObjectsOf eval:usesBenchmarkRelease ;
    sh:class eval:BenchmarkRelease ;
    sh:message "eval:usesBenchmarkRelease must point to an eval:BenchmarkRelease." .

eval:HasBenchmarkReleaseSubjectShape
    a sh:NodeShape ;
    sh:targetSubjectsOf eval:hasBenchmarkRelease ;
    sh:class eval:BenchmarkSuite ;
    sh:message "eval:hasBenchmarkRelease must have an eval:BenchmarkSuite as subject." .

eval:HasBenchmarkReleaseObjectShape
    a sh:NodeShape ;
    sh:targetObjectsOf eval:hasBenchmarkRelease ;
    sh:class eval:BenchmarkRelease ;
    sh:message "eval:hasBenchmarkRelease must point to an eval:BenchmarkRelease." .

eval:HasTaskSubjectShape
    a sh:NodeShape ;
    sh:targetSubjectsOf eval:hasTask ;
    sh:class eval:BenchmarkRelease ;
    sh:message "eval:hasTask must have an eval:BenchmarkRelease as subject." .


eval:TaskUsesDatasetSubjectShape
    a sh:NodeShape ;
    sh:targetSubjectsOf eval:usesDataset ;
    sh:node eval:TaskClassShape ;
    sh:message "eval:usesDataset must have an eval:Task as subject." .

eval:TaskUsesDatasetObjectShape
    a sh:NodeShape ;
    sh:targetObjectsOf eval:usesDataset ;
    sh:node data:DatasetClassShape ;
    sh:message "eval:usesDataset must point to a data:Dataset." .

eval:HasMetricResultSubjectShape
    a sh:NodeShape ;
    sh:targetSubjectsOf eval:hasMetricResult ;
    sh:class eval:BenchmarkResult ;
    sh:message "eval:hasMetricResult must have an eval:BenchmarkResult as subject." .

eval:HasMetricResultObjectShape
    a sh:NodeShape ;
    sh:targetObjectsOf eval:hasMetricResult ;
    sh:class eval:MetricResult ;
    sh:message "eval:hasMetricResult must point to an eval:MetricResult." .

eval:HasResultTaskSubjectShape
    a sh:NodeShape ;
    sh:targetSubjectsOf eval:hasResultTask ;
    sh:class eval:BenchmarkResult ;
    sh:message "eval:hasResultTask must have an eval:BenchmarkResult as subject." .

eval:HasResultTaskObjectShape
    a sh:NodeShape ;
    sh:targetObjectsOf eval:hasResultTask ;
    sh:node eval:TaskClassShape ;
    sh:message "eval:hasResultTask must point to an eval:Task." .

eval:ProducesBenchmarkResultSubjectShape
    a sh:NodeShape ;
    sh:targetSubjectsOf eval:producesBenchmarkResult ;
    sh:class eval:MachineLearningEvaluationRun ;
    sh:message "eval:producesBenchmarkResult must have an eval:MachineLearningEvaluationRun as subject." .

eval:ProducesBenchmarkResultObjectShape
    a sh:NodeShape ;
    sh:targetObjectsOf eval:producesBenchmarkResult ;
    sh:class eval:BenchmarkResult ;
    sh:message "eval:producesBenchmarkResult must point to an eval:BenchmarkResult." .

eval:HasMetricTypeSubjectShape
    a sh:NodeShape ;
    sh:targetSubjectsOf eval:hasMetricType ;
    sh:class eval:MetricResult ;
    sh:message "eval:hasMetricType must have an eval:MetricResult as subject." .

eval:HasMetricTypeObjectShape
    a sh:NodeShape ;
    sh:targetObjectsOf eval:hasMetricType ;
    sh:class eval:MetricType ;
    sh:message "eval:hasMetricType must point to an eval:MetricType." .

eval:HasSplitSubjectShape
    a sh:NodeShape ;
    sh:targetSubjectsOf eval:hasSplit ;
    sh:class eval:Fold ;
    sh:message "eval:hasSplit must have an eval:Fold as subject." .

eval:HasSplitObjectShape
    a sh:NodeShape ;
    sh:targetObjectsOf eval:hasSplit ;
    sh:node data:DatasetClassShape ;
    sh:message "eval:hasSplit must point to a data:Dataset or data:DatasetSplit." .

eval:HasTrainSplitSubjectShape
    a sh:NodeShape ;
    sh:targetSubjectsOf eval:hasTrainSplit ;
    sh:class eval:Fold ;
    sh:message "eval:hasTrainSplit must have an eval:Fold as subject." .

eval:HasTrainSplitObjectShape
    a sh:NodeShape ;
    sh:targetObjectsOf eval:hasTrainSplit ;
    sh:node data:DatasetClassShape ;
    sh:message "eval:hasTrainSplit must point to a data:Dataset or data:DatasetSplit." .

eval:HasValidationSplitSubjectShape
    a sh:NodeShape ;
    sh:targetSubjectsOf eval:hasValidationSplit ;
    sh:class eval:Fold ;
    sh:message "eval:hasValidationSplit must have an eval:Fold as subject." .

eval:HasValidationSplitObjectShape
    a sh:NodeShape ;
    sh:targetObjectsOf eval:hasValidationSplit ;
    sh:node data:DatasetClassShape ;
    sh:message "eval:hasValidationSplit must point to a data:Dataset or data:DatasetSplit." .

eval:HasTestSplitSubjectShape
    a sh:NodeShape ;
    sh:targetSubjectsOf eval:hasTestSplit ;
    sh:class eval:Fold ;
    sh:message "eval:hasTestSplit must have an eval:Fold as subject." .

eval:HasTestSplitObjectShape
    a sh:NodeShape ;
    sh:targetObjectsOf eval:hasTestSplit ;
    sh:node data:DatasetClassShape ;
    sh:message "eval:hasTestSplit must point to a data:Dataset or data:DatasetSplit." .

eval:ProducesMetricResultSubjectShape
    a sh:NodeShape ;
    sh:targetSubjectsOf eval:producesMetricResult ;
    sh:class eval:Fold ;
    sh:message "eval:producesMetricResult must have an eval:Fold as subject." .

eval:ProducesMetricResultObjectShape
    a sh:NodeShape ;
    sh:targetObjectsOf eval:producesMetricResult ;
    sh:class eval:MetricResult ;
    sh:message "eval:producesMetricResult must point to an eval:MetricResult." .

eval:MachineLearningEvaluationRunShape
    a sh:NodeShape ;
    sh:targetClass eval:MachineLearningEvaluationRun ;
    sh:property [
        sh:path eval:evaluatesModelVariant ;
        sh:minCount 1 ;
        sh:maxCount 1 ;
        sh:node arch:ModelVariantClassShape ;
        sh:message "every eval:MachineLearningEvaluationRun should evaluate exactly one model variant." ;
    ] ;
    sh:property [
        sh:path eval:evaluatesTask ;
        sh:minCount 1 ;
        sh:maxCount 1 ;
        sh:node eval:TaskClassShape ;
        sh:message "every eval:MachineLearningEvaluationRun should evaluate exactly one task." ;
    ] ;
    sh:property [
        sh:path eval:usesBenchmarkRelease ;
        sh:maxCount 1 ;
        sh:class eval:BenchmarkRelease ;
        sh:message "eval:usesBenchmarkRelease should have at most one benchmark release per evaluation run." ;
    ] ;
    sh:property [
        sh:path eval:producesBenchmarkResult ;
        sh:minCount 1 ;
        sh:class eval:BenchmarkResult ;
        sh:message "every eval:MachineLearningEvaluationRun should produce at least one benchmark result." ;
    ] .


eval:BenchmarkSuiteShape
    a sh:NodeShape ;
    sh:targetClass eval:BenchmarkSuite ;
    sh:property [
        sh:path eval:hasBenchmarkRelease ;
        sh:minCount 1 ;
        sh:class eval:BenchmarkRelease ;
        sh:message "every eval:BenchmarkSuite should declare at least one benchmark release." ;
    ] .

eval:BenchmarkResultShape
    a sh:NodeShape ;
    sh:targetClass eval:BenchmarkResult ;
    sh:property [
        sh:path eval:hasMetricResult ;
        sh:minCount 1 ;
        sh:class eval:MetricResult ;
        sh:message "every eval:BenchmarkResult should contain at least one metric result." ;
    ] ;
    sh:property [
        sh:path eval:hasResultTask ;
        sh:minCount 1 ;
        sh:maxCount 1 ;
        sh:node eval:TaskClassShape ;
        sh:message "every eval:BenchmarkResult should be attached to exactly one task." ;
    ] .

eval:MetricResultShape
    a sh:NodeShape ;
    sh:targetClass eval:MetricResult ;
    sh:property [
        sh:path eval:hasMetricType ;
        sh:minCount 1 ;
        sh:maxCount 1 ;
        sh:class eval:MetricType ;
        sh:message "every eval:MetricResult should have exactly one metric type." ;
    ] ;
    sh:property [
        sh:path eval:hasMetricValue ;
        sh:minCount 1 ;
        sh:maxCount 1 ;
        sh:or (
            [ sh:datatype xsd:float ]
            [ sh:datatype xsd:double ]
            [ sh:datatype xsd:decimal ]
            [ sh:datatype xsd:integer ]
        ) ;
        sh:message "every eval:MetricResult should have exactly one numeric metric value." ;
    ] .

eval:FoldShape
    a sh:NodeShape ;
    sh:targetClass eval:Fold ;
    sh:property [
        sh:path eval:hasIndex ;
        sh:maxCount 1 ;
        sh:datatype xsd:integer ;
        sh:minInclusive 0 ;
        sh:message "eval:hasIndex should be a non-negative integer and have at most one value per fold." ;
    ] .

eval:EvaluatesTaskFunctionalShape
    a sh:NodeShape ;
    sh:targetSubjectsOf eval:evaluatesTask ;
    sh:property [
        sh:path eval:evaluatesTask ;
        sh:maxCount 1 ;
        sh:message "eval:evaluatesTask is functional and must have at most one value." ;
    ] .

eval:UsesBenchmarkReleaseFunctionalShape
    a sh:NodeShape ;
    sh:targetSubjectsOf eval:usesBenchmarkRelease ;
    sh:property [
        sh:path eval:usesBenchmarkRelease ;
        sh:maxCount 1 ;
        sh:message "eval:usesBenchmarkRelease should have at most one value per evaluation run." ;
    ] .

eval:MetricTypeFunctionalShape
    a sh:NodeShape ;
    sh:targetSubjectsOf eval:hasMetricType ;
    sh:property [
        sh:path eval:hasMetricType ;
        sh:maxCount 1 ;
        sh:message "eval:hasMetricType should have at most one value per metric result." ;
    ] .

eval:MetricValueFunctionalShape
    a sh:NodeShape ;
    sh:targetSubjectsOf eval:hasMetricValue ;
    sh:property [
        sh:path eval:hasMetricValue ;
        sh:maxCount 1 ;
        sh:message "eval:hasMetricValue should have at most one value per metric result." ;
    ] .

eval:FoldIndexFunctionalShape
    a sh:NodeShape ;
    sh:targetSubjectsOf eval:hasIndex ;
    sh:property [
        sh:path eval:hasIndex ;
        sh:maxCount 1 ;
        sh:message "eval:hasIndex should have at most one value per fold." ;
    ] .

eval:MetricValueDatatypeShape
    a sh:NodeShape ;
    sh:targetSubjectsOf eval:hasMetricValue ;
    sh:property [
        sh:path eval:hasMetricValue ;
        sh:or (
            [ sh:datatype xsd:float ]
            [ sh:datatype xsd:double ]
            [ sh:datatype xsd:decimal ]
            [ sh:datatype xsd:integer ]
        ) ;
        sh:message "eval:hasMetricValue must be numeric." ;
    ] .

eval:FoldIndexDatatypeShape
    a sh:NodeShape ;
    sh:targetSubjectsOf eval:hasIndex ;
    sh:property [
        sh:path eval:hasIndex ;
        sh:datatype xsd:integer ;
        sh:minInclusive 0 ;
        sh:message "eval:hasIndex must be a non-negative integer." ;
    ] .


# Architecture domain/range rules.

arch:HasVariantSubjectShape
    a sh:NodeShape ;
    sh:targetSubjectsOf arch:hasVariant ;
    sh:class arch:ModelFamily ;
    sh:message "arch:hasVariant must have an arch:ModelFamily as subject." .

arch:HasVariantObjectShape
    a sh:NodeShape ;
    sh:targetObjectsOf arch:hasVariant ;
    sh:class arch:ModelVariant ;
    sh:message "arch:hasVariant must point to an arch:ModelVariant." .

arch:HasArchitectureSubjectShape
    a sh:NodeShape ;
    sh:targetSubjectsOf arch:hasMachineLearningArchitecture ;
    sh:class arch:ModelVariant ;
    sh:message "arch:hasMachineLearningArchitecture must have an arch:ModelVariant as subject." .

arch:HasArchitectureObjectShape
    a sh:NodeShape ;
    sh:targetObjectsOf arch:hasMachineLearningArchitecture ;
    sh:node arch:MachineLearningArchitectureClassShape ;
    sh:message "arch:hasMachineLearningArchitecture must point to an arch:MachineLearningArchitecture." .

arch:HasArchitectureConfigurationSubjectShape
    a sh:NodeShape ;
    sh:targetSubjectsOf arch:hasMachineLearningArchitectureConfiguration ;
    sh:node arch:MachineLearningArchitectureClassShape ;
    sh:message "arch:hasMachineLearningArchitectureConfiguration must have an arch:MachineLearningArchitecture as subject." .

arch:HasArchitectureConfigurationObjectShape
    a sh:NodeShape ;
    sh:targetObjectsOf arch:hasMachineLearningArchitectureConfiguration ;
    sh:class arch:MachineLearningArchitectureConfiguration ;
    sh:message "arch:hasMachineLearningArchitectureConfiguration must point to an arch:MachineLearningArchitectureConfiguration." .

arch:HasComponentSubjectShape
    a sh:NodeShape ;
    sh:targetSubjectsOf arch:hasComponent ;
    sh:or (
        [ sh:node arch:MachineLearningArchitectureClassShape ]
        [ sh:class arch:ModelVariant ]
    ) ;
    sh:message "arch:hasComponent must have an arch:MachineLearningArchitecture or arch:ModelVariant as subject." .

arch:HasComponentObjectShape
    a sh:NodeShape ;
    sh:targetObjectsOf arch:hasComponent ;
    sh:node arch:MachineLearningArchitectureClassShape ;
    sh:message "arch:hasComponent must point to an arch:MachineLearningArchitecture." .

arch:HasBackboneArchitectureSubjectShape
    a sh:NodeShape ;
    sh:targetSubjectsOf arch:hasBackboneArchitecture ;
    sh:or (
        [ sh:node arch:MachineLearningArchitectureClassShape ]
        [ sh:class arch:ModelVariant ]
    ) ;
    sh:message "arch:hasBackboneArchitecture must have an arch:MachineLearningArchitecture or arch:ModelVariant as subject." .

arch:HasBackboneArchitectureObjectShape
    a sh:NodeShape ;
    sh:targetObjectsOf arch:hasBackboneArchitecture ;
    sh:node arch:MachineLearningArchitectureClassShape ;
    sh:message "arch:hasBackboneArchitecture must point to an arch:MachineLearningArchitecture." .

arch:HasHeadArchitectureSubjectShape
    a sh:NodeShape ;
    sh:targetSubjectsOf arch:hasHeadArchitecture ;
    sh:or (
        [ sh:node arch:MachineLearningArchitectureClassShape ]
        [ sh:class arch:ModelVariant ]
    ) ;
    sh:message "arch:hasHeadArchitecture must have an arch:MachineLearningArchitecture or arch:ModelVariant as subject." .

arch:HasHeadArchitectureObjectShape
    a sh:NodeShape ;
    sh:targetObjectsOf arch:hasHeadArchitecture ;
    sh:node arch:MachineLearningArchitectureClassShape ;
    sh:message "arch:hasHeadArchitecture must point to an arch:MachineLearningArchitecture." .


# Training domain/range rules.

train:GeneratedBySubjectShape
    a sh:NodeShape ;
    sh:targetSubjectsOf train:generatedBy ;
    sh:node train:ModelMaterializationClassShape ;
    sh:message "train:generatedBy must have a train:ModelMaterialization as subject." .

train:GeneratedByObjectShape
    a sh:NodeShape ;
    sh:targetObjectsOf train:generatedBy ;
    sh:node train:TrainingRunClassShape ;
    sh:message "train:generatedBy must point to a train:MachineLearningTrainingRun." .

train:HasHyperparameterValueSubjectShape
    a sh:NodeShape ;
    sh:targetSubjectsOf train:hasHyperparameterValue ;
    sh:node train:TrainingRunClassShape ;
    sh:message "train:hasHyperparameterValue must have a train:MachineLearningTrainingRun as subject." .

train:HasHyperparameterValueObjectShape
    a sh:NodeShape ;
    sh:targetObjectsOf train:hasHyperparameterValue ;
    sh:class train:MachineLearningHyperparameterValue ;
    sh:message "train:hasHyperparameterValue must point to a train:MachineLearningHyperparameterValue." .

train:OfHyperparameterSubjectShape
    a sh:NodeShape ;
    sh:targetSubjectsOf train:ofHyperparameter ;
    sh:class train:MachineLearningHyperparameterValue ;
    sh:message "train:ofHyperparameter must have a train:MachineLearningHyperparameterValue as subject." .

train:OfHyperparameterObjectShape
    a sh:NodeShape ;
    sh:targetObjectsOf train:ofHyperparameter ;
    sh:class train:MachineLearningHyperparameter ;
    sh:message "train:ofHyperparameter must point to a train:MachineLearningHyperparameter." .

train:TrainsModelVariantSubjectShape
    a sh:NodeShape ;
    sh:targetSubjectsOf train:trainsModelVariant ;
    sh:node train:TrainingRunClassShape ;
    sh:message "train:trainsModelVariant must have a train:MachineLearningTrainingRun as subject." .

train:TrainsModelVariantObjectShape
    a sh:NodeShape ;
    sh:targetObjectsOf train:trainsModelVariant ;
    sh:node arch:ModelVariantClassShape ;
    sh:message "train:trainsModelVariant must point to an arch:ModelVariant." .

train:UsesDatasetSubjectShape
    a sh:NodeShape ;
    sh:targetSubjectsOf train:usesDataset ;
    sh:node train:TrainingRunClassShape ;
    sh:message "train:usesDataset must have a MachineLearningTrainingRun as subject." .

train:UsesDatasetObjectShape
    a sh:NodeShape ;
    sh:targetObjectsOf train:usesDataset ;
    sh:node data:DatasetClassShape ;
    sh:message "train:usesDataset must point to a data:Dataset." .

train:UsesOptimizerSubjectShape
    a sh:NodeShape ;
    sh:targetSubjectsOf train:usesOptimizer ;
    sh:node train:TrainingRunClassShape ;
    sh:message "train:usesOptimizer must have a train:MachineLearningTrainingRun as subject." .

train:UsesOptimizerObjectShape
    a sh:NodeShape ;
    sh:targetObjectsOf train:usesOptimizer ;
    sh:class train:Optimizer ;
    sh:message "train:usesOptimizer must point to a train:Optimizer." .

train:UsesLossSubjectShape
    a sh:NodeShape ;
    sh:targetSubjectsOf train:usesLoss ;
    sh:node train:TrainingRunClassShape ;
    sh:message "train:usesLoss must have a train:MachineLearningTrainingRun as subject." .

train:UsesLossObjectShape
    a sh:NodeShape ;
    sh:targetObjectsOf train:usesLoss ;
    sh:class train:LossFunction ;
    sh:message "train:usesLoss must point to a train:LossFunction." .

train:UsesObjectiveFunctionSubjectShape
    a sh:NodeShape ;
    sh:targetSubjectsOf train:usesObjectiveFunction ;
    sh:node train:TrainingRunClassShape ;
    sh:message "train:usesObjectiveFunction must have a train:MachineLearningTrainingRun as subject." .

train:UsesObjectiveFunctionObjectShape
    a sh:NodeShape ;
    sh:targetObjectsOf train:usesObjectiveFunction ;
    sh:node train:ObjectiveFunctionClassShape ;
    sh:message "train:usesObjectiveFunction must point to a train:ObjectiveFunction." .

train:UsesSamplingMethodSubjectShape
    a sh:NodeShape ;
    sh:targetSubjectsOf train:usesSamplingMethod ;
    sh:node train:TrainingRunClassShape ;
    sh:message "train:usesSamplingMethod must have a train:MachineLearningTrainingRun as subject." .

train:UsesSamplingMethodObjectShape
    a sh:NodeShape ;
    sh:targetObjectsOf train:usesSamplingMethod ;
    sh:class train:SamplingMethod ;
    sh:message "train:usesSamplingMethod must point to a train:SamplingMethod." .

train:HasStudentModelSubjectShape
    a sh:NodeShape ;
    sh:targetSubjectsOf train:hasStudentModel ;
    sh:class train:MachineLearningDistillationRun ;
    sh:message "train:hasStudentModel must have a train:MachineLearningDistillationRun as subject." .

train:HasStudentModelObjectShape
    a sh:NodeShape ;
    sh:targetObjectsOf train:hasStudentModel ;
    sh:class train:ModelMaterialization ;
    sh:message "train:hasStudentModel must point to a train:ModelMaterialization." .

train:HasTeacherModelSubjectShape
    a sh:NodeShape ;
    sh:targetSubjectsOf train:hasTeacherModel ;
    sh:class train:MachineLearningDistillationRun ;
    sh:message "train:hasTeacherModel must have a train:MachineLearningDistillationRun as subject." .

train:HasTeacherModelObjectShape
    a sh:NodeShape ;
    sh:targetObjectsOf train:hasTeacherModel ;
    sh:class train:ModelMaterialization ;
    sh:message "train:hasTeacherModel must point to a train:ModelMaterialization." .

train:InitializedFromModelMaterializationSubjectShape
    a sh:NodeShape ;
    sh:targetSubjectsOf train:initializedFromModelMaterialization ;
    sh:class train:MachineLearningAdaptationRun ;
    sh:message "train:initializedFromModelMaterialization must have a train:MachineLearningAdaptationRun as subject." .

train:InitializedFromModelMaterializationObjectShape
    a sh:NodeShape ;
    sh:targetObjectsOf train:initializedFromModelMaterialization ;
    sh:class train:ModelMaterialization ;
    sh:message "train:initializedFromModelMaterialization must point to a train:ModelMaterialization." .


# Dataset domain/range rules.

data:HasDataRepresentationSubjectShape
    a sh:NodeShape ;
    sh:targetSubjectsOf data:hasDataRepresentation ;
    sh:node data:DatasetClassShape ;
    sh:message "data:hasDataRepresentation must have a data:Dataset as subject." .

data:HasDataRepresentationObjectShape
    a sh:NodeShape ;
    sh:targetObjectsOf data:hasDataRepresentation ;
    sh:class data:DataRepresentation ;
    sh:message "data:hasDataRepresentation must point to a data:DataRepresentation." .

data:HasLabellingMethodSubjectShape
    a sh:NodeShape ;
    sh:targetSubjectsOf data:hasLabellingMethod ;
    sh:class data:LabeledDataset ;
    sh:message "data:hasLabellingMethod must have a data:LabeledDataset as subject." .

data:HasLabellingMethodObjectShape
    a sh:NodeShape ;
    sh:targetObjectsOf data:hasLabellingMethod ;
    sh:class data:LabellingMethod ;
    sh:message "data:hasLabellingMethod must point to a data:LabellingMethod." .

data:UsesAttributeSubjectShape
    a sh:NodeShape ;
    sh:targetSubjectsOf data:usesAttribute ;
    sh:node data:DatasetClassShape ;
    sh:message "data:usesAttribute must have a data:Dataset as subject." .

data:UsesAttributeObjectShape
    a sh:NodeShape ;
    sh:targetObjectsOf data:usesAttribute ;
    sh:class data:DatasetAttribute ;
    sh:message "data:usesAttribute must point to a data:DatasetAttribute." .

data:UsesAttributeAsInputFeatureSubjectShape
    a sh:NodeShape ;
    sh:targetSubjectsOf data:usesAttributeAsInputFeature ;
    sh:node data:TrainingDatasetClassShape ;
    sh:message "data:usesAttributeAsInputFeature must have a train:TrainingDataset as subject." .

data:UsesAttributeAsInputFeatureObjectShape
    a sh:NodeShape ;
    sh:targetObjectsOf data:usesAttributeAsInputFeature ;
    sh:class data:DatasetAttribute ;
    sh:message "data:usesAttributeAsInputFeature must point to a data:DatasetAttribute." .

data:UsesAttributeAsTargetFeatureSubjectShape
    a sh:NodeShape ;
    sh:targetSubjectsOf data:usesAttributeAsTargetFeature ;
    sh:class data:LabeledDataset ;
    sh:message "data:usesAttributeAsTargetFeature must have a data:LabeledDataset as subject." .

data:UsesAttributeAsTargetFeatureObjectShape
    a sh:NodeShape ;
    sh:targetObjectsOf data:usesAttributeAsTargetFeature ;
    sh:class data:DatasetAttribute ;
    sh:message "data:usesAttributeAsTargetFeature must point to a data:DatasetAttribute." .

data:WasDerivedFromDatasetSubjectShape
    a sh:NodeShape ;
    sh:targetSubjectsOf data:wasDerivedFromDataset ;
    sh:node data:DatasetClassShape ;
    sh:message "data:wasDerivedFromDataset must have a data:Dataset as subject." .

data:WasDerivedFromDatasetObjectShape
    a sh:NodeShape ;
    sh:targetObjectsOf data:wasDerivedFromDataset ;
    sh:node data:DatasetClassShape ;
    sh:message "data:wasDerivedFromDataset must point to a data:Dataset." .


# Minimal structural rules.

arch:ModelFamilyShape
    a sh:NodeShape ;
    sh:targetClass arch:ModelFamily ;
    sh:property [
        sh:path arch:hasVariant ;
        sh:minCount 1 ;
        sh:message "every arch:ModelFamily should link to at least one arch:ModelVariant." ;
    ] .

arch:ModelVariantShape
    a sh:NodeShape ;
    sh:targetClass arch:ModelVariant ;
    sh:property [
        sh:path arch:hasMachineLearningArchitecture ;
        sh:minCount 1 ;
        sh:maxCount 1 ;
        sh:node arch:MachineLearningArchitectureClassShape ;
        sh:message "every arch:ModelVariant should link to exactly one arch:MachineLearningArchitecture." ;
    ] ;
    sh:property [
        sh:path arch:hasParameterNumber ;
        sh:maxCount 1 ;
        sh:datatype xsd:integer ;
        sh:minInclusive 0 ;
        sh:message "arch:hasParameterNumber on a ModelVariant must be at most one non-negative integer." ;
    ] .

arch:MachineLearningArchitectureShape
    a sh:NodeShape ;
    sh:targetClass arch:MachineLearningArchitecture ;
    sh:property [
        sh:path arch:hasMachineLearningArchitectureConfiguration ;
        sh:minCount 1 ;
        sh:maxCount 1 ;
        sh:class arch:MachineLearningArchitectureConfiguration ;
        sh:message "every arch:MachineLearningArchitecture should link to exactly one arch:MachineLearningArchitectureConfiguration." ;
    ] .

arch:MachineLearningArchitectureSubclassShape
    a sh:NodeShape ;
    sh:targetClass arch:NeuralNetworkArchitecture,
        arch:AttentionBasedArchitecture,
        arch:EquivariantGraphNeuralNetworkArchitecture,
        arch:EquivariantMessagePassingNeuralNetworkArchitecture,
        arch:EquivariantNeuralNetworkArchitecture,
        arch:GatedNeuralNetworkArchitecture,
        arch:GraphConvolutionArchitecture,
        arch:GraphNeuralNetworkArchitecture,
        arch:GraphTransformerArchitecture,
        arch:MessagePassingNeuralNetworkArchitecture,
        arch:TransformerArchitecture ;
    sh:property [
        sh:path arch:hasMachineLearningArchitectureConfiguration ;
        sh:minCount 1 ;
        sh:maxCount 1 ;
        sh:class arch:MachineLearningArchitectureConfiguration ;
        sh:severity sh:Warning ;
        sh:message "every generated architecture subclass individual should link to exactly one architecture configuration." ;
    ] .

train:TrainingRunShape
    a sh:NodeShape ;
    sh:targetClass train:MachineLearningTrainingRun ;
    sh:property [
        sh:path train:trainsModelVariant ;
        sh:minCount 1 ;
        sh:node arch:ModelVariantClassShape ;
        sh:message "every train:MachineLearningTrainingRun should link to at least one trained arch:ModelVariant." ;
    ] .

train:HyperparameterValueShape
    a sh:NodeShape ;
    sh:targetClass train:MachineLearningHyperparameterValue ;
    sh:property [
        sh:path train:ofHyperparameter ;
        sh:minCount 1 ;
        sh:maxCount 1 ;
        sh:class train:MachineLearningHyperparameter ;
        sh:message "every train:MachineLearningHyperparameterValue should link to exactly one train:MachineLearningHyperparameter." ;
    ] .

train:CheckpointShape
    a sh:NodeShape ;
    sh:targetClass train:Checkpoint ;
    sh:property [
        sh:path train:generatedBy ;
        sh:minCount 1 ;
        sh:maxCount 1 ;
        sh:node train:TrainingRunClassShape ;
        sh:message "every train:Checkpoint should be generated by exactly one training run." ;
    ] .

train:PretrainingRunShape
    a sh:NodeShape ;
    sh:targetClass train:PretrainingRun ;
    sh:property [
        sh:path train:trainsModelVariant ;
        sh:minCount 1 ;
        sh:node arch:ModelVariantClassShape ;
        sh:message "every train:PretrainingRun should link to at least one trained model variant." ;
    ] ;
    sh:property [
        sh:path train:usesDataset ;
        sh:minCount 1 ;
        sh:node data:DatasetClassShape ;
        sh:message "every train:PretrainingRun should use at least one dataset." ;
    ] .

data:LabeledDatasetShape
    a sh:NodeShape ;
    sh:targetClass data:LabeledDataset ;
    sh:property [
        sh:path data:hasDataRepresentation ;
        sh:minCount 1 ;
        sh:maxCount 1 ;
        sh:class data:DataRepresentation ;
        sh:message "every data:LabeledDataset should have exactly one data representation." ;
    ] ;
    sh:property [
        sh:path data:hasLabellingMethod ;
        sh:minCount 1 ;
        sh:maxCount 1 ;
        sh:class data:LabellingMethod ;
        sh:message "every data:LabeledDataset should have exactly one labelling method." ;
    ] ;
    sh:property [
        sh:path data:usesAttributeAsInputFeature ;
        sh:minCount 1 ;
        sh:class data:DatasetAttribute ;
        sh:severity sh:Warning ;
        sh:message "every data:LabeledDataset should declare at least one input feature." ;
    ] ;
    sh:property [
        sh:path data:usesAttributeAsTargetFeature ;
        sh:minCount 1 ;
        sh:class data:DatasetAttribute ;
        sh:message "every data:LabeledDataset should declare at least one target feature." ;
    ] .

data:DatasetSplitShape
    a sh:NodeShape ;
    sh:targetClass data:DatasetSplit ;
    sh:property [
        sh:path data:hasDataRepresentation ;
        sh:minCount 1 ;
        sh:maxCount 1 ;
        sh:class data:DataRepresentation ;
        sh:message "every data:DatasetSplit should have exactly one data representation." ;
    ] ;
    sh:property [
        sh:path data:wasDerivedFromDataset ;
        sh:minCount 1 ;
        sh:maxCount 1 ;
        sh:node data:DatasetClassShape ;
        sh:message "every data:DatasetSplit should be derived from exactly one dataset." ;
    ] .

# Functional-property cardinality rules.

arch:ArchitectureConfigurationFunctionalShape
    a sh:NodeShape ;
    sh:targetSubjectsOf arch:hasMachineLearningArchitectureConfiguration ;
    sh:property [
        sh:path arch:hasMachineLearningArchitectureConfiguration ;
        sh:maxCount 1 ;
        sh:message "arch:hasMachineLearningArchitectureConfiguration is functional and must have at most one value." ;
    ] .

arch:ParameterNumberFunctionalShape
    a sh:NodeShape ;
    sh:targetSubjectsOf arch:hasParameterNumber ;
    sh:property [
        sh:path arch:hasParameterNumber ;
        sh:maxCount 1 ;
        sh:message "arch:hasParameterNumber is functional and must have at most one value per subject." ;
    ] .

arch:LayerCountFunctionalShape
    a sh:NodeShape ;
    sh:targetSubjectsOf arch:hasLayerCount ;
    sh:property [
        sh:path arch:hasLayerCount ;
        sh:maxCount 1 ;
        sh:message "arch:hasLayerCount should have at most one value per architecture configuration." ;
    ] .

arch:HiddenDimensionFunctionalShape
    a sh:NodeShape ;
    sh:targetSubjectsOf arch:hasHiddenDimension ;
    sh:property [
        sh:path arch:hasHiddenDimension ;
        sh:maxCount 1 ;
        sh:message "arch:hasHiddenDimension should have at most one value per architecture configuration." ;
    ] .

data:DataRepresentationFunctionalShape
    a sh:NodeShape ;
    sh:targetSubjectsOf data:hasDataRepresentation ;
    sh:property [
        sh:path data:hasDataRepresentation ;
        sh:maxCount 1 ;
        sh:message "data:hasDataRepresentation is functional and must have at most one value." ;
    ] .

data:LabellingMethodFunctionalShape
    a sh:NodeShape ;
    sh:targetSubjectsOf data:hasLabellingMethod ;
    sh:property [
        sh:path data:hasLabellingMethod ;
        sh:maxCount 1 ;
        sh:message "data:hasLabellingMethod should have at most one value per dataset." ;
    ] .

data:WasDerivedFromDatasetFunctionalShape
    a sh:NodeShape ;
    sh:targetSubjectsOf data:wasDerivedFromDataset ;
    sh:property [
        sh:path data:wasDerivedFromDataset ;
        sh:maxCount 1 ;
        sh:message "data:wasDerivedFromDataset should have at most one value per dataset in this KG." ;
    ] .

data:NumberOfSamplesFunctionalShape
    a sh:NodeShape ;
    sh:targetSubjectsOf data:hasNumberOfSamples ;
    sh:property [
        sh:path data:hasNumberOfSamples ;
        sh:maxCount 1 ;
        sh:message "data:hasNumberOfSamples should have at most one value per dataset." ;
    ] .

train:TeacherModelFunctionalShape
    a sh:NodeShape ;
    sh:targetSubjectsOf train:hasTeacherModel ;
    sh:property [
        sh:path train:hasTeacherModel ;
        sh:maxCount 1 ;
        sh:message "train:hasTeacherModel is functional and must have at most one value." ;
    ] .

train:InitializedFromModelMaterializationFunctionalShape
    a sh:NodeShape ;
    sh:targetSubjectsOf train:initializedFromModelMaterialization ;
    sh:property [
        sh:path train:initializedFromModelMaterialization ;
        sh:maxCount 1 ;
        sh:message "train:initializedFromModelMaterialization is functional and must have at most one value." ;
    ] .

# Datatype rules.

arch:ParameterNumberShape
    a sh:NodeShape ;
    sh:targetSubjectsOf arch:hasParameterNumber ;
    sh:property [
        sh:path arch:hasParameterNumber ;
        sh:datatype xsd:integer ;
        sh:minInclusive 0 ;
        sh:message "arch:hasParameterNumber must be a non-negative integer." ;
    ] .

arch:LayerCountShape
    a sh:NodeShape ;
    sh:targetSubjectsOf arch:hasLayerCount ;
    sh:property [
        sh:path arch:hasLayerCount ;
        sh:datatype xsd:integer ;
        sh:minInclusive 0 ;
        sh:message "arch:hasLayerCount must be a non-negative integer." ;
    ] .

arch:HiddenDimensionShape
    a sh:NodeShape ;
    sh:targetSubjectsOf arch:hasHiddenDimension ;
    sh:property [
        sh:path arch:hasHiddenDimension ;
        sh:datatype xsd:integer ;
        sh:minInclusive 1 ;
        sh:message "arch:hasHiddenDimension must be a positive integer." ;
    ] .

data:NumberOfSamplesShape
    a sh:NodeShape ;
    sh:targetSubjectsOf data:hasNumberOfSamples ;
    sh:property [
        sh:path data:hasNumberOfSamples ;
        sh:datatype xsd:integer ;
        sh:minInclusive 0 ;
        sh:message "data:hasNumberOfSamples must be a non-negative integer." ;
    ] .

train:UsesBatchedSelectionShape
    a sh:NodeShape ;
    sh:targetSubjectsOf train:usesBatchedSelection ;
    sh:property [
        sh:path train:usesBatchedSelection ;
        sh:datatype xsd:boolean ;
        sh:message "train:usesBatchedSelection must be an xsd:boolean." ;
    ] .

arch:MachineLearningArchitectureShape
  a sh:NodeShape ;
  sh:targetClass arch:MachineLearningArchitecture ;
  sh:property [
    sh:path arch:hasMachineLearningArchitectureConfiguration ;
    sh:minCount 1 ;
    sh:maxCount 1 ;
    sh:class arch:MachineLearningArchitectureConfiguration ;
  ] .

train:CheckpointShape
  a sh:NodeShape ;
  sh:targetClass train:Checkpoint ;
  sh:property [
    sh:path train:generatedBy ;
    sh:minCount 1 ;
    sh:node train:TrainingRunClassShape ;
  ] .

"""


def strip_rdf_star_lines(ttl_text):
    return "\n".join(
        line for line in ttl_text.splitlines()
        if not line.lstrip().startswith("<<")
    )


def parse_turtle_file(path, strip_rdf_star=False):
    graph = Graph()
    text = path.read_text(encoding="utf-8")

    if strip_rdf_star:
        text = strip_rdf_star_lines(text)

    graph.parse(data=text, format="turtle", publicID=path.as_uri())
    return graph


def load_ontology_graph():
    graph = Graph()

    for path in ONTOLOGY_FILES:
        if path.exists():
            graph.parse(path, format="turtle")

    return graph


def load_shapes_graph():
    graph = Graph()
    graph.parse(data=SHACL_SHAPES, format="turtle")
    return graph


def known_ontology_classes(ontology_graph):
    classes = set()

    for class_type in (OWL.Class, RDFS.Class):
        classes.update(ontology_graph.subjects(RDF.type, class_type))

    return classes


def find_unknown_ontology_types(data_graph, ontology_graph):
    known_classes = known_ontology_classes(ontology_graph)
    unknown_types = set()

    for _, _, rdf_type in data_graph.triples((None, RDF.type, None)):
        if not isinstance(rdf_type, URIRef):
            continue

        rdf_type_text = str(rdf_type)

        if rdf_type == OWL.NamedIndividual:
            continue

        if rdf_type_text.startswith(ONTOLOGY_NAMESPACES) and rdf_type not in known_classes:
            unknown_types.add(rdf_type)

    return sorted(unknown_types, key=str)

def get_shacl_report(ttl_path):
    ttl_path = Path(ttl_path)

    data_graph = parse_turtle_file(ttl_path, strip_rdf_star=True)
    ontology_graph = load_ontology_graph()
    shapes_graph = load_shapes_graph()

    conforms, report_graph, report_text = validate(
        data_graph=data_graph,
        shacl_graph=shapes_graph,
        ont_graph=None,
        inference="none",
        abort_on_first=False,
        allow_infos=True,
        allow_warnings=True,
        meta_shacl=False,
        advanced=True,
        serialize_report_graph=False,
    )

    unknown_types = find_unknown_ontology_types(
        data_graph,
        ontology_graph,
    )

    return {
        "ttl_path": str(ttl_path),
        "conforms": conforms and not unknown_types,
        "shacl_conforms": conforms,
        "report_graph": report_graph,
        "report_text": report_text,
        "unknown_types": [str(x) for x in unknown_types],
    }

def validate_ttl_file(ttl_path, report_dir=None):
    print(f"\n=== VALIDATE SHACL ===")
    print(f"TTL: {ttl_path}")

    data_graph = parse_turtle_file(ttl_path, strip_rdf_star=True)
    ontology_graph = load_ontology_graph()
    shapes_graph = load_shapes_graph()

    conforms, report_graph, report_text = validate(
        data_graph=data_graph,
        shacl_graph=shapes_graph,
        ont_graph=None,
        inference="none",
        abort_on_first=False,
        allow_infos=True,
        allow_warnings=True,
        meta_shacl=False,
        advanced=True,
        serialize_report_graph=False,
    )

    unknown_types = find_unknown_ontology_types(data_graph, ontology_graph)

    if report_dir:
        report_dir = Path(report_dir)
        report_dir.mkdir(parents=True, exist_ok=True)

        report_txt_path = report_dir / f"{ttl_path.stem}_shacl_report.txt"
        report_txt_path.write_text(report_text, encoding="utf-8")

        report_ttl_path = report_dir / f"{ttl_path.stem}_shacl_report.ttl"
        report_graph.serialize(destination=report_ttl_path, format="turtle")

    if conforms and not unknown_types:
        print("[OK] SHACL conforms")
        return True

    if not conforms:
        print("[FAIL] SHACL validation failed")
        print(report_text)

    if unknown_types:
        print("[FAIL] Unknown ontology classes used as rdf:type:")
        for rdf_type in unknown_types:
            print(f"  - {rdf_type}")

    return False


def ttl_files_from_args(args):
    if args.ttl:
        return [Path(args.ttl).resolve()]

    env_ttl = args.env_ttl
    if env_ttl:
        return [Path(env_ttl).resolve()]

    ttl_dir = Path(args.ttl_dir)

    return sorted(
        ttl_path.resolve() for ttl_path in ttl_dir.glob("*.ttl")
        if not ttl_path.name.startswith("__")
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Validate generated Matbench ontology TTL files with SHACL."
    )
    parser.add_argument(
        "--ttl",
        help="Path to one generated TTL file. Defaults to INPUT_TTL_FILE or outputs/ttl/*.ttl.",
    )
    parser.add_argument(
        "--ttl-dir",
        default=str(DEFAULT_TTL_DIR),
        help="Directory containing TTL files to validate.",
    )
    parser.add_argument(
        "--env-ttl",
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
    "--report-dir",
    default=str(BASE_DIR / "outputs" / "shacl_reports"),
    help="Directory where SHACL reports are written.",
    )   
    return parser.parse_args()


def main():
    args = parse_args()

    if args.env_ttl is None:
        import os
        args.env_ttl = os.getenv("INPUT_TTL_FILE")

    ttl_files = ttl_files_from_args(args)

    if not ttl_files:
        print(f"No TTL files found in {args.ttl_dir}")
        return 1

    if not args.ttl and not args.env_ttl:
        print(f"Validating {len(ttl_files)} TTL file(s) from {args.ttl_dir}")

    failed = []

    for ttl_path in ttl_files:
        try:
            if not validate_ttl_file(ttl_path, report_dir=args.report_dir):
                failed.append(ttl_path)
        except Exception as error:
            print(f"\n[ERROR] {ttl_path}")
            print(error)
            failed.append(ttl_path)

    if failed:
        print(f"\nValidation failed for {len(failed)} file(s).")
        return 1

    print(f"\nValidated {len(ttl_files)} file(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
