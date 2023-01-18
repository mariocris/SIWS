package com.mycompany.project;

import java.io.IOException;
import java.util.List;
import de.uni_leipzig.dbs.pprl.primat.common.csv.CSVWriter;
import de.uni_leipzig.dbs.pprl.primat.common.model.NamedRecordSchemaConfiguration;
import de.uni_leipzig.dbs.pprl.primat.common.model.Record;
import de.uni_leipzig.dbs.pprl.primat.common.model.attributes.NonQidAttributeType;
import de.uni_leipzig.dbs.pprl.primat.common.model.attributes.QidAttributeType;
import de.uni_leipzig.dbs.pprl.primat.common.utils.DatasetReader;
import de.uni_leipzig.dbs.pprl.primat.dataowner.preprocessing.FieldNormalizer;
import de.uni_leipzig.dbs.pprl.primat.dataowner.preprocessing.FieldSplitter;
import de.uni_leipzig.dbs.pprl.primat.dataowner.preprocessing.NormalizeDefinition;
import de.uni_leipzig.dbs.pprl.primat.dataowner.preprocessing.PartySupplier;
import de.uni_leipzig.dbs.pprl.primat.dataowner.preprocessing.SplitDefinition;
import de.uni_leipzig.dbs.pprl.primat.dataowner.preprocessing.normalizing.AccentRemover;
import de.uni_leipzig.dbs.pprl.primat.dataowner.preprocessing.normalizing.LetterLowerCaseToNumberNormalizer;
import de.uni_leipzig.dbs.pprl.primat.dataowner.preprocessing.normalizing.LetterUpperCaseToNumberNormalizer;
import de.uni_leipzig.dbs.pprl.primat.dataowner.preprocessing.normalizing.LowerCaseNormalizer;
import de.uni_leipzig.dbs.pprl.primat.dataowner.preprocessing.normalizing.NormalizerChain;
import de.uni_leipzig.dbs.pprl.primat.dataowner.preprocessing.normalizing.SpecialCharacterRemover;
import de.uni_leipzig.dbs.pprl.primat.dataowner.preprocessing.normalizing.SubstringNormalizer;
import de.uni_leipzig.dbs.pprl.primat.dataowner.preprocessing.normalizing.TrimNormalizer;
import de.uni_leipzig.dbs.pprl.primat.dataowner.preprocessing.normalizing.UmlautNormalizer;
import de.uni_leipzig.dbs.pprl.primat.dataowner.preprocessing.splitting.BlankSplitter;
import de.uni_leipzig.dbs.pprl.primat.dataowner.preprocessing.splitting.DotSplitter;
import java.io.IOException;
import java.util.List;

import de.uni_leipzig.dbs.pprl.primat.common.csv.CSVWriter;
import de.uni_leipzig.dbs.pprl.primat.common.extraction.FeatureExtractor;
import de.uni_leipzig.dbs.pprl.primat.common.extraction.qgram.BigramExtractor;
import de.uni_leipzig.dbs.pprl.primat.common.model.NamedRecordSchemaConfiguration;
import de.uni_leipzig.dbs.pprl.primat.common.model.Record;
import de.uni_leipzig.dbs.pprl.primat.common.model.attributes.NonQidAttributeType;
import de.uni_leipzig.dbs.pprl.primat.common.model.attributes.QidAttributeType;
import de.uni_leipzig.dbs.pprl.primat.common.utils.DatasetReader;
import de.uni_leipzig.dbs.pprl.primat.common.utils.RandomFactory;
import de.uni_leipzig.dbs.pprl.primat.dataowner.encoding.Encoder;
import de.uni_leipzig.dbs.pprl.primat.dataowner.encoding.bloomfilter.BloomFilterDefinition;
import de.uni_leipzig.dbs.pprl.primat.dataowner.encoding.bloomfilter.BloomFilterEncoder;
import de.uni_leipzig.dbs.pprl.primat.dataowner.encoding.bloomfilter.BloomFilterExtractorDefinition;
import de.uni_leipzig.dbs.pprl.primat.dataowner.encoding.bloomfilter.hardening.NoHardener;
import de.uni_leipzig.dbs.pprl.primat.dataowner.encoding.bloomfilter.hashing.HashingMethod;
import de.uni_leipzig.dbs.pprl.primat.dataowner.encoding.bloomfilter.hashing.RandomHashing;

import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVPrinter;

import java.io.IOException;
import java.util.Collection;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

import de.uni_leipzig.dbs.pprl.primat.common.extraction.lsh.LshKeyGenerator;
import de.uni_leipzig.dbs.pprl.primat.common.extraction.lsh.RandomHammingLshKeyGenerator;
import de.uni_leipzig.dbs.pprl.primat.common.model.LinkageConstraint;
import de.uni_leipzig.dbs.pprl.primat.common.model.NamedRecordSchemaConfiguration;
import de.uni_leipzig.dbs.pprl.primat.common.model.Party;
import de.uni_leipzig.dbs.pprl.primat.common.model.PartyPair;
import de.uni_leipzig.dbs.pprl.primat.common.model.Record;
import de.uni_leipzig.dbs.pprl.primat.common.model.attributes.NonQidAttributeType;
import de.uni_leipzig.dbs.pprl.primat.common.model.attributes.QidAttributeType;
import de.uni_leipzig.dbs.pprl.primat.common.utils.DatasetReader;
import de.uni_leipzig.dbs.pprl.primat.common.utils.DoubleListAggregator;
import de.uni_leipzig.dbs.pprl.primat.lu.blocking.Blocker;
import de.uni_leipzig.dbs.pprl.primat.lu.blocking.lsh.LshBlocker;
import de.uni_leipzig.dbs.pprl.primat.lu.classification.Classificator;
import de.uni_leipzig.dbs.pprl.primat.lu.classification.ThresholdClassificator;
import de.uni_leipzig.dbs.pprl.primat.lu.evaluation.QualityEvaluator;
import de.uni_leipzig.dbs.pprl.primat.lu.evaluation.QualityMetrics;
import de.uni_leipzig.dbs.pprl.primat.lu.evaluation.true_match_checker.IdEqualityTrueMatchChecker;
import de.uni_leipzig.dbs.pprl.primat.lu.evaluation.true_match_checker.TrueMatchChecker;
import de.uni_leipzig.dbs.pprl.primat.lu.linkage_result.LinkageResult;
import de.uni_leipzig.dbs.pprl.primat.lu.linkage_result.LinkageResultPartition;
import de.uni_leipzig.dbs.pprl.primat.lu.linkage_result.LinkageResultPartitionFactory;
import de.uni_leipzig.dbs.pprl.primat.lu.linkage_result.matches.MatchStrategyFactory;
import de.uni_leipzig.dbs.pprl.primat.lu.linkage_result.matches.SimilarityGraphMatchStrategyFactory;
import de.uni_leipzig.dbs.pprl.primat.lu.linkage_result.non_matches.IgnoreNonMatchesStrategyFactory;
import de.uni_leipzig.dbs.pprl.primat.lu.linkage_result.non_matches.NonMatchStrategyFactory;
import de.uni_leipzig.dbs.pprl.primat.lu.matching.Matcher;
import de.uni_leipzig.dbs.pprl.primat.lu.matching.batch.BatchMatcher;
import de.uni_leipzig.dbs.pprl.primat.lu.postprocessing.NoPostprocessor;
import de.uni_leipzig.dbs.pprl.primat.lu.postprocessing.PostprocessingStrategy;
import de.uni_leipzig.dbs.pprl.primat.lu.postprocessing.best_match.MaxBothPostprocessor;
import de.uni_leipzig.dbs.pprl.primat.lu.postprocessing.best_match.MaxRightPostprocessor;
import de.uni_leipzig.dbs.pprl.primat.lu.similarity_calculation.attribute_similarity.BitSetAttributeSimilarityCalculator;
import de.uni_leipzig.dbs.pprl.primat.lu.similarity_calculation.record_similarity.BaseRecordSimilarityCalculator;
import de.uni_leipzig.dbs.pprl.primat.lu.similarity_calculation.record_similarity.RecordSimilarityCalculator;
import de.uni_leipzig.dbs.pprl.primat.lu.similarity_classification.BatchSimilarityClassification;
import de.uni_leipzig.dbs.pprl.primat.lu.similarity_classification.ComparisonStrategy;
import de.uni_leipzig.dbs.pprl.primat.lu.similarity_classification.RedundancyCheckStrategy;
import de.uni_leipzig.dbs.pprl.primat.lu.similarity_classification.SimilarityClassification;
import de.uni_leipzig.dbs.pprl.primat.lu.similarity_function.binary.BinarySimilarity;
import de.uni_leipzig.dbs.pprl.primat.lu.similarity_vector.BaseSimilarityVectorAggregator;
import de.uni_leipzig.dbs.pprl.primat.lu.similarity_vector.BaseSimilarityVectorFlattener;
import de.uni_leipzig.dbs.pprl.primat.lu.similarity_vector.FlatSimilarityVectorAggregator;
import de.uni_leipzig.dbs.pprl.primat.lu.similarity_vector.SimilarityVectorAggregator;
import de.uni_leipzig.dbs.pprl.primat.lu.similarity_vector.SimilarityVectorFlattener;
import de.uni_leipzig.dbs.pprl.primat.lu.utils.StandardThresholdClassificationRefinement;
import de.uni_leipzig.dbs.pprl.primat.lu.utils.ThresholdClassificationRefinement;
import java.io.BufferedReader;

import java.io.FileReader;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.Arrays;


public class Project {
	
    public static void DataOwners(String inputhPathName, String partyName) throws IOException {
        
        // Dataset Reading

        final NamedRecordSchemaConfiguration rsc = new NamedRecordSchemaConfiguration.Builder()
                        .add(0, NonQidAttributeType.ID)
                        .add(1, NonQidAttributeType.PARTY) 
                        .add(2, QidAttributeType.STRING, "fname")
                        .add(3, QidAttributeType.STRING, "gname")
                        .add(4, QidAttributeType.STRING, "address")
                        .add(5, QidAttributeType.STRING, "dob")
                        .build();	

        final String inputPath = inputhPathName + ".csv"; 
        final String outputPath = inputhPathName + "_encoded.csv";

        final DatasetReader reader = new DatasetReader(inputPath, true, rsc);
        final List<Record> records = reader.read();


        // Data Cleaning

        final PartySupplier partySupp = new PartySupplier();
        partySupp.preprocess(records);

        final SplitDefinition splitDef = new SplitDefinition();
        splitDef.setSplitter("address", new BlankSplitter(2));
        splitDef.setSplitter("dob", new DotSplitter(3));

        final FieldSplitter fs = new FieldSplitter(splitDef);
        fs.preprocess(records);

        final NormalizerChain normChain = new NormalizerChain(
                List.of(new UmlautNormalizer(), new TrimNormalizer(), new LowerCaseNormalizer(), new AccentRemover(),
                        new SpecialCharacterRemover(), new SubstringNormalizer(0, 12)));

        final NormalizerChain numberNorm = new NormalizerChain(new LetterLowerCaseToNumberNormalizer(),
                new LetterUpperCaseToNumberNormalizer());

        final NormalizeDefinition normDef = new NormalizeDefinition();
        normDef.setNormalizer(0, normChain);
        normDef.setNormalizer(1, normChain);
        normDef.setNormalizer(2, numberNorm);
        normDef.setNormalizer(3, normChain);
        normDef.setNormalizer(4, numberNorm);
        normDef.setNormalizer(5, numberNorm);
        normDef.setNormalizer(6, numberNorm);

        final FieldNormalizer fn = new FieldNormalizer(normDef);
        fn.preprocess(records);


        // Encoding

        final FeatureExtractor featEx = new BigramExtractor(true);
        final int k = 10;

        final BloomFilterExtractorDefinition exDef = new BloomFilterExtractorDefinition();
        exDef.setColumns(0,1,2,3,4,5,6);
        exDef.setExtractors(featEx);
        exDef.setNumberOfHashFunctions(k);

        final HashingMethod hashing = new RandomHashing(1024, RandomFactory.SECURE_RANDOM);

        final BloomFilterDefinition def1 = new BloomFilterDefinition();
        def1.setName("BS");
        def1.setBfLength(1024);
        def1.setHashingMethod(hashing);
        def1.setFeatureExtractors(List.of(exDef));
        def1.setHardener(new NoHardener());

        final Encoder encoder = new BloomFilterEncoder(List.of(def1));

        final List<Record> encodedRecords = encoder.encode(records);

        
        // Final dataset writing

        final CSVWriter csvWriter = new CSVWriter(outputPath);
        csvWriter.writeRecords(encodedRecords, encoder.getSchema());
        
        
        List<List<String>> newRecords = new ArrayList<>();
        try (BufferedReader br = new BufferedReader(new FileReader(outputPath))) {
            String line;
            while ((line = br.readLine()) != null) {
                String[] values = line.split(",");
                newRecords.add(Arrays.asList(values));
            }
        }
        
        newRecords.remove(0);
        
        for(List<String> r: newRecords){
            r.set(2, partyName);
        }
        
        FileWriter csvWriterNew = new FileWriter(outputPath);
        csvWriterNew.append("ID");
        csvWriterNew.append(",");
        csvWriterNew.append("GID");
        csvWriterNew.append(",");
        csvWriterNew.append("PARTY");
        csvWriterNew.append(",");
        csvWriterNew.append("BS");
        csvWriterNew.append("\n");

        for (List<String> rowData : newRecords) {
            csvWriterNew.append(String.join(",", rowData));
            csvWriterNew.append("\n");
        }

        csvWriterNew.flush();
        csvWriterNew.close();
        
    }
    
    public static void MergeCSV(String pathNameA, String pathNameB, String outputPath) throws IOException {
        
        String pathA = pathNameA + "_encoded.csv";
        String pathB = pathNameB + "_encoded.csv";

        
        List<List<String>> recordsA = new ArrayList<>();
        try (BufferedReader br = new BufferedReader(new FileReader(pathA))) {
            String line;
            while ((line = br.readLine()) != null) {
                String[] values = line.split(",");
                recordsA.add(Arrays.asList(values));
            }
        }
        
        recordsA.remove(0);

        
        List<List<String>> recordsB = new ArrayList<>();
        try (BufferedReader br = new BufferedReader(new FileReader(pathB))) {
            String line;
            while ((line = br.readLine()) != null) {
                String[] values = line.split(",");
                recordsB.add(Arrays.asList(values));
            }
        }
        
        recordsB.remove(0);
        
        
        int n = recordsA.size();
        for(List<String> r: recordsB){
            r.set(0, Integer.toString(n));
            n++;
        }
        
        recordsA.addAll(recordsB);
        

        FileWriter csvWriterNew = new FileWriter(outputPath);
        csvWriterNew.append("ID");
        csvWriterNew.append(",");
        csvWriterNew.append("GID");
        csvWriterNew.append(",");
        csvWriterNew.append("PARTY");
        csvWriterNew.append(",");
        csvWriterNew.append("BS");
        csvWriterNew.append("\n");

        for (List<String> rowData : recordsA) {
            csvWriterNew.append(String.join(",", rowData));
            csvWriterNew.append("\n");
        }

        csvWriterNew.flush();
        csvWriterNew.close();
        
    }
    
    public static void LinkageUnit(String inputhPathName) throws IOException {
        
        // Dataset Reading
        
        // Schema Configuration
        final NamedRecordSchemaConfiguration rsc = new NamedRecordSchemaConfiguration.Builder()
				.add(0, NonQidAttributeType.ID)
				.add(1, NonQidAttributeType.GLOBAL_ID)
				.add(2, NonQidAttributeType.PARTY)
				//.add(2, QidAttributeType.BITSET, "BS")
                                //.add(2, QidAttributeType.STRING, "S1")
                                .add(3, QidAttributeType.BITSET, "BS")
                                //.add(3, QidAttributeType.STRING, "S1")
				.build();	
        System.out.println("#Named Record Schema Configuration rsc " 
                + rsc.getNonQidAttributeMap() 
                + rsc.getQidAttributeNameMap());
        
        // Parameters
        final double threshold = 0.8; 
        final int matches = 20; 

        final String namePartyA = "A";
        final String namePartyB = "B";

        // Read the dataset file
        final DatasetReader reader = new DatasetReader(inputhPathName, true, rsc);
        final List<Record> dataset = reader.read();
        System.out.println("#dataset " + dataset);

        // List the records for the 2 dataset A and B
        final List<Record> datasetA = dataset.stream()
                .filter(r -> r.getPartyAttribute().getValue().getName().equals(namePartyA)).collect(Collectors.toList());
        System.out.println("#dataset A " + datasetA);

        final List<Record> datasetB = dataset.stream()
                .filter(r -> r.getPartyAttribute().getValue().getName().equals(namePartyB)).collect(Collectors.toList());
        System.out.println("#dataset B " + datasetB);

        // Create input for the 2 dataset
        final Map<Party, Collection<Record>> input = new HashMap<>();
        input.put(new Party(namePartyA), datasetA);
        input.put(new Party(namePartyB), datasetB);
        System.out.println("#input " + input);

        System.out.println("#Records source " + namePartyA + ": " + datasetA.size());
        System.out.println("#Records source " + namePartyB + ": " + datasetB.size());

        final ComparisonStrategy compStrat = ComparisonStrategy.SOURCE_CONSISTENT;

        final LshKeyGenerator keyGen = new RandomHammingLshKeyGenerator(16, 30, 1024, 42L);
        final Blocker blocker = new LshBlocker(keyGen);

        final RecordSimilarityCalculator simCalc = new BaseRecordSimilarityCalculator(
                List.of(new BitSetAttributeSimilarityCalculator(List.of(BinarySimilarity.JACCARD_SIMILARITY))));

        final SimilarityVectorFlattener flattener = new BaseSimilarityVectorFlattener(
                List.of(DoubleListAggregator.FIRST));
        final FlatSimilarityVectorAggregator aggregator = new BaseSimilarityVectorAggregator(
                DoubleListAggregator.FIRST);
        final SimilarityVectorAggregator agg = new SimilarityVectorAggregator(flattener, aggregator);


        final Classificator classifier = new ThresholdClassificator(threshold, agg);

        final MatchStrategyFactory<Record> matchFactory = new SimilarityGraphMatchStrategyFactory<>();
        final NonMatchStrategyFactory<Record> nonMatchFactory = new IgnoreNonMatchesStrategyFactory<>();
        final LinkageResultPartitionFactory<Record> linkResFac = new LinkageResultPartitionFactory<>(matchFactory, nonMatchFactory);
        final SimilarityClassification simClass = new BatchSimilarityClassification(compStrat, simCalc, classifier,
                RedundancyCheckStrategy.MATCH_TWICE, linkResFac);
        final ThresholdClassificationRefinement threshRef = new StandardThresholdClassificationRefinement();		

        final PostprocessingStrategy<Record> postprocessor = new PostprocessingStrategy<>();
        postprocessor.setPostprocessor(LinkageConstraint.ONE_TO_ONE, new MaxBothPostprocessor<Record>());
        postprocessor.setPostprocessor(LinkageConstraint.MANY_TO_ONE, new MaxRightPostprocessor<Record>());
        postprocessor.setPostprocessor(LinkageConstraint.ONE_TO_MANY, new NoPostprocessor<Record>());
        postprocessor.setPostprocessor(LinkageConstraint.MANY_TO_MANY, new NoPostprocessor<Record>());

        final Matcher<Record> matcher = new BatchMatcher(blocker, simClass, threshRef, postprocessor);

        final LinkageResult<Record> linkRes = matcher.match(input);

        final TrueMatchChecker trueMatchChecker = new IdEqualityTrueMatchChecker();
        final QualityEvaluator<Record> evaluator = new QualityEvaluator<>(trueMatchChecker);

        final PartyPair partyPairAB = new PartyPair(new Party(namePartyA), new Party(namePartyB));

        final LinkageResultPartition<Record> part = linkRes.getPartition(partyPairAB);
        evaluator.addMatches(part.getMatchStrategy().getMatches());


        final long truePos = evaluator.getTruePositives();
        final long falsePos = evaluator.getFalsePositives();

        final double recall = QualityMetrics.getRecall(truePos, matches);
        final double precision = QualityMetrics.getPrecision(truePos, truePos + falsePos);
        final double fmeasure = QualityMetrics.getFMeasure(recall, precision);

        System.out.println("Recall: " + recall);
        System.out.println("Precision: " + precision);
        System.out.println("F-Measuer: " + fmeasure);

        
    }
    
    public static void main(String[] args) throws IOException {

        String pathA = "ds1_a";
        String pathB = "ds1_b";
        String outputPath = "ds1_encoded.csv";
        
//        DataOwners(pathA, "A");
//        DataOwners(pathB, "B");
//        
//        MergeCSV(pathA, pathB, outputPath);
        
        LinkageUnit(outputPath);
    }
}
