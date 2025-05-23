package org.air2025;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

import org.apache.lucene.analysis.CharArraySet;
import org.apache.lucene.analysis.WordlistLoader;
import org.apache.lucene.analysis.fr.FrenchAnalyzer;
import org.apache.lucene.document.*;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.util.BytesRef;

import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Iterator;
import java.util.stream.Stream;

public class IndexBuilder {
    /**
     * Contains a static method to index French JSON-s
     */
    private static class FrenchJsonLuceneIndexer {
        /**
         * For indexing the French text JSONs
         * @param inputFilePath JSON path
         * @param outputFilePath Index path
         * @param customStopWordListPath if there is any custom stop-word list to be used
         * @throws IOException if files cannot be read
         */
        public static void buildIndex(String inputFilePath, String outputFilePath, String customStopWordListPath) throws IOException {
            Directory dir = FSDirectory.open(Paths.get(outputFilePath));
            FrenchAnalyzer analyzer = null;
            if(customStopWordListPath != null){
                try{
                    CharArraySet stopwords = WordlistLoader.getWordSet(
                        new FileReader(Path.of(customStopWordListPath).toFile()));
                    analyzer = new FrenchAnalyzer(stopwords);
                }
                catch(Exception exception){
                    System.out.println("Custom stopword list could not be loaded because of the following exception:" 
                    + exception.getMessage() +"The in-built list will be used!");
                    analyzer = new FrenchAnalyzer();
                }

            }
            else{
                analyzer = new FrenchAnalyzer();
            }



            IndexWriterConfig config = new IndexWriterConfig(analyzer);
            config.setOpenMode(IndexWriterConfig.OpenMode.CREATE);

            IndexWriter writer = new IndexWriter(dir, config);
            ObjectMapper mapper = new ObjectMapper();

            try (Stream<Path> paths = Files.walk(Paths.get(inputFilePath))) {
                paths.filter(Files::isRegularFile).forEach(path -> {
                    try {
                        JsonNode json = mapper.readTree(path.toFile());

                        for(JsonNode jsonNode : json) {
                            String id = jsonNode.has("id") ? jsonNode.get("id").asText() : path.getFileName().toString();
                            String body = jsonNode.has("contents") ? jsonNode.get("contents").asText() : "";

                            Document doc = new Document();
                            // Add ID properly
                            doc.add(new StringField("id", id, Field.Store.YES)); // Stored and indexed
                            doc.add(new BinaryDocValuesField("id", new BytesRef(id))); // Required for Pyserini

                            doc.add(new TextField("contents", body, Field.Store.YES)); // Pyserini reads "contents"

                            writer.addDocument(doc);

                        }
                        System.out.println("Indexed: " + path.getFileName().toString());


                    } catch (IOException e) {
                        System.err.println("Failed to index " + path + ": " + e.getMessage());
                    }
                });
            }

            writer.close();
            System.out.println("sAll documents indexed to: " + outputFilePath);
        }
    }
    public static void main(String[] args) {
        if (args.length > 3 || args.length < 2) {
            System.err.println("Usage: java IndexBuilder <input_json_path> <output_index_path> (optional) <custom_stopword_list_file_path> Remember to specify absolute paths!");
            System.exit(1);
        }

        String inputDataPath = args[0];
        String outputIndexPath = args[1];

        // Check input data path
        if (!Files.exists(Paths.get(inputDataPath))) {
            System.err.println("Input data does not exist: " + inputDataPath);
            System.exit(1);
        }

        // Check output data path, try to create if it does not exist
        if (!Files.exists(Paths.get(outputIndexPath))) {
            if(new File(outputIndexPath).mkdirs()){
                System.err.println("Output index path does not exist AND couldn't be created! " + outputIndexPath);
            }
        }

        // Check custom stopwords file (if specified)
        String stopWordListPath = null;
        if (args.length == 3){
            stopWordListPath = args[2];
            if (!Files.exists(Paths.get(stopWordListPath))){
                System.err.println("File with custom stopword list does not exist: " + inputDataPath + ". The in-built list will be used.");
                System.exit(1);
            }
        }

        // Now the indexing part
        try{
            FrenchJsonLuceneIndexer.buildIndex(inputDataPath, outputIndexPath, stopWordListPath);
            System.out.println("Index built successfully!");
        }
        catch(Exception e) {
            System.err.println("Error building index: " + e.getMessage());
        }

    }
}