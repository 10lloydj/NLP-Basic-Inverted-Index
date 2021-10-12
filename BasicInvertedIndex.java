/**
 * Basic Inverted Index
 * 
 * This Map Reduce program should build an Inverted Index from a set of files.
 * Each token (the key) in a given file should reference the file it was found 
 * in. 
 * 
 * The output of the program should look like this:
 * sometoken [file001= tfidf, file002= tfidf, ... ]
 * 
 * 
 */
package uk.ac.man.cs.comp38211.exercise;

import java.io.*;
import java.util.*;
import java.lang.Math;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.*;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.FileSplit;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.cli.OptionBuilder;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;
import org.apache.log4j.Logger;

import uk.ac.man.cs.comp38211.io.array.ArrayListOfLongsWritable;
import uk.ac.man.cs.comp38211.io.array.ArrayListWritable;
import uk.ac.man.cs.comp38211.io.map.HMapSFW;
import uk.ac.man.cs.comp38211.io.map.HashMapWritable;
import uk.ac.man.cs.comp38211.io.pair.PairOfStrings;
import uk.ac.man.cs.comp38211.io.pair.PairOfWritables;
import uk.ac.man.cs.comp38211.ir.Stemmer;
import uk.ac.man.cs.comp38211.ir.StopAnalyser;
import uk.ac.man.cs.comp38211.util.XParser;

public class BasicInvertedIndex extends Configured implements Tool
{
    private static final Logger LOG = Logger
            .getLogger(BasicInvertedIndex.class);


    public static class Map extends 
            Mapper<Object, Text, Text, Text>
    {
    	
        // INPUTFILE holds the name of the current file
        private final static Text INPUTFILE = new Text();
        
        // TOKEN should be set to the current token rather than creating a 
        // new Text object for each one
        @SuppressWarnings("unused")
        private final static Text TOKEN = new Text();

        // The StopAnalyser class helps remove stop words
        @SuppressWarnings("unused")
        private StopAnalyser stopAnalyser = new StopAnalyser();
        
        // The stem method wraps the functionality of the Stemmer
        // class, which trims extra characters from English words
        // Please refer to the Stemmer class for more comments
        @SuppressWarnings("unused")
        private String stem(String word)
        {
            Stemmer s = new Stemmer();

            // A char[] word is added to the stemmer with its length,
            // then stemmed
            s.add(word.toCharArray(), word.length());
            s.stem();

            // return the stemmed char[] word as a string
            return s.toString();
        }
        
        // This method gets the name of the file the current Mapper is working
        // on
        @Override
        public void setup(Context context)
        {
            String inputFilePath = ((FileSplit) context.getInputSplit()).getPath().toString();
            String[] pathComponents = inputFilePath.split("/");
            INPUTFILE.set(pathComponents[pathComponents.length - 1]);
        }
         
        // TODO
        // This Mapper should read in a line, convert it to a set of tokens
        // and output each token with the name of the file it was found in
        public void map(Object key, Text value, Context context)
                throws IOException, InterruptedException
        {
        	
        	// in-mapper aggregation hashmap
        	// unsure of parameters, is it text and text?
        	// left needs to be text as its the token, right will be the value of an integer
        	// I've decided to leave out the in-mapper as I've developed the tf-idf
        	// It will complicate the input
        	// HashMapWritable<Text,Text> localCache = new HashMapWritable<Text, Text>();
        	
            // The line is broken up and turned into an iterator 
            String line = value.toString();
            StringTokenizer itr = new StringTokenizer(line);

            // note: performance is 50 secs with this in, without 64 seconds
           // adds the stopwords to a hash set
            Set<String> stopWordSet = new HashSet<String>();
            for(String word : stopWords) {
            	stopWordSet.add(word);
            } // for  
            
            while (itr.hasMoreTokens())
            {
            	String token = itr.nextToken();
            	           	
            	// makes the token lowercase
            	token = token.toLowerCase();
            	// removes hyphens 
            	token.replaceAll("\u2014","" );
            	// remove non-alphabetic characters
            	token = token.replaceAll(("[^A-Za-z0-9 ]"), "");
            	
            	//if hashset of stopwords contains the current token, remove 
            	if(stopWordSet.contains(token))
            		token = token.replaceAll(token, "");
            	
            	// stopAnalyser removes stopwords            	
            	if(StopAnalyser.isStopWord(token))
            		token = token.replaceAll(token, "");
            	
            	// stemmed word
            	// stemmer messes up the word on some such as genuis, results in genui
            	String stemmedtoken = stem(token);
            	TOKEN.set(stemmedtoken);
                           	
            	context.write(TOKEN, INPUTFILE);
            } // while
        } // void map
    } // class Map

    // changed the parameter to allow HMapSFW
    // HMapSFW is Key = type String, Value= type float
    public static class Reduce extends Reducer<Text, Text, Text, HMapSFW>
    {
        // TODO
        // This Reduce Job should take in a key and an iterable of file names
        // It should convert this iterable to a writable array list and output
        // it along with the key
    	
    	// This Reduce Job instead of an array list, uses a map to allow tfidf to be ouput
        public void reduce(
                Text key,
                Iterable<Text> values,
                Context context) throws IOException, InterruptedException
        {
        	// for each of the values in the input, adds it to a map as a key
            Iterator<Text> iter = values.iterator();
            // map used to track the term frequency
            HMapSFW docfreq = new HMapSFW();
            // map used to store tfidf calculations
            HMapSFW calctf = new HMapSFW();
            float count = 0, df = 0, tf = 0, logtf = 0, idf = 0, tfidf = 0;
            
            // iterates through the file list counting the frequency in each document of the token
            // uses this frequency for the term frequency 
            while(iter.hasNext()) {            
            	Text currentFile = iter.next();
            	String currentString = currentFile.toString();
            	
            	// if the token exists in the document, increments the count
            	if(docfreq.containsKey(currentString)) {
            		count = docfreq.get(currentString)+1;
            		docfreq.put(currentString, count);
            	} // if
            	else {
            		// adds the new word with the first count
            		docfreq.put(currentString, 1 );
            	} // else           	
            	
            	// tfidf calculations
            	df = (float) docfreq.size();
                idf = (float) Math.log10(6/df);
                
                tf = docfreq.get(currentString);
            	logtf = (float) Math.log10(1+tf);

            	tfidf = logtf*idf;     
            	// round results to 2 decimal places
            	tfidf =(float)(Math.round(tfidf * 100d) / 100d);
            	
            	calctf.put(currentString, tfidf);
            } // while
                      
                
            // emits the word and a map with the files and ftf-idf 
            
            context.write(key, calctf);
        } // void reduce
    } // class Reduce

    // Lets create an object! :)
    public BasicInvertedIndex()
    {
    }

    // Variables to hold cmd line args
    private static final String INPUT = "input";
    private static final String OUTPUT = "output";
    private static final String NUM_REDUCERS = "numReducers";

    @SuppressWarnings({ "static-access" })
    public int run(String[] args) throws Exception
    {
        
        // Handle command line args
        Options options = new Options();
        options.addOption(OptionBuilder.withArgName("path").hasArg()
                .withDescription("input path").create(INPUT));
        options.addOption(OptionBuilder.withArgName("path").hasArg()
                .withDescription("output path").create(OUTPUT));
        options.addOption(OptionBuilder.withArgName("num").hasArg()
                .withDescription("number of reducers").create(NUM_REDUCERS));

        CommandLine cmdline = null;
        CommandLineParser parser = new XParser(true);

        try
        {
            cmdline = parser.parse(options, args);
        }
        catch (ParseException exp)
        {
            System.err.println("Error parsing command line: "
                    + exp.getMessage());
            System.err.println(cmdline);
            return -1;
        }

        // If we are missing the input or output flag, let the user know
        if (!cmdline.hasOption(INPUT) || !cmdline.hasOption(OUTPUT))
        {
            System.out.println("args: " + Arrays.toString(args));
            HelpFormatter formatter = new HelpFormatter();
            formatter.setWidth(120);
            formatter.printHelp(this.getClass().getName(), options);
            ToolRunner.printGenericCommandUsage(System.out);
            return -1;
        }

        // Create a new Map Reduce Job
        Configuration conf = new Configuration();
        Job job = new Job(conf);
        String inputPath = cmdline.getOptionValue(INPUT);
        String outputPath = cmdline.getOptionValue(OUTPUT);
        int reduceTasks = cmdline.hasOption(NUM_REDUCERS) ? Integer
                .parseInt(cmdline.getOptionValue(NUM_REDUCERS)) : 1;

        // Set the name of the Job and the class it is in
        job.setJobName("Basic Inverted Index");
        job.setJarByClass(BasicInvertedIndex.class);
        job.setNumReduceTasks(reduceTasks);
        
        // Set the Mapper and Reducer class (no need for combiner here)
        job.setMapperClass(Map.class);
        job.setReducerClass(Reduce.class);
        
        // Set the Output Classes
        job.setMapOutputKeyClass(Text.class);
        job.setMapOutputValueClass(Text.class);
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(ArrayListWritable.class);

        // Set the input and output file paths
        FileInputFormat.setInputPaths(job, new Path(inputPath));
        FileOutputFormat.setOutputPath(job, new Path(outputPath));
        
        // Time the job whilst it is running
        long startTime = System.currentTimeMillis();
        job.waitForCompletion(true);
        LOG.info("Job Finished in " + (System.currentTimeMillis() - startTime)
                / 1000.0 + " seconds");

        // Returning 0 lets everyone know the job was successful
        return 0;
    }

    public static void main(String[] args) throws Exception
    {
        ToolRunner.run(new BasicInvertedIndex(), args);
    }
    
    // stop word list found at http://xpo6.com/list-of-english-stop-words/
    public static String[] stopWords = {"a", "about", "above", "above", "across",
			"after", "afterwards", "again", "against", 
			"all", "almost", "alone", "along", "already",
			"also","although","always","am","among", "amongst",
			"amoungst", "amount",  "an", "and", "another", "any",
			"anyhow","anyone","anything","anyway", "anywhere",
			"are", "around", "as",  "at", "back","be","became",
			"because","become","becomes", "becoming", "been", 
			"before", "beforehand", "behind", "being", "below", 
			"beside", "besides", "between", "beyond", "bill", 
			"both", "bottom","but", "by", "call", "can", 
			"cannot", "cant", "co", "con", "could", "couldnt", 
			"cry", "de", "describe", "detail", "do", "done", 
			"down", "due", "during", "each", "eg", "eight",
			"either", "eleven","else", "elsewhere", "empty", "enough",
			"etc", "even", "ever", "every", "everyone", "everything", 
			"everywhere", "except", "few", "fifteen", "fify", "fill",
			"find", "fire", "first", "five", "for", "former", "formerly",
			"forty", "found", "four", "from", "front", "full", "further",
			"get", "give", "go", "had", "has", "hasnt", "have", "he", 
			"hence", "her", "here", "hereafter", "hereby", "herein", 
			"hereupon", "hers", "herself", "him", "himself", "his",
			"how", "however", "hundred", "ie", "if", "in", "inc", 
			"indeed", "interest", "into", "is", "it", "its", "itself",
			"keep", "last", "latter", "latterly", "least", "less", "ltd",
			"made", "many", "may", "me", "meanwhile", "might", "mill",
			"mine", "more", "moreover", "most", "mostly", "move", "much",
			"must", "my", "myself", "name", "namely", "neither", "never", 
			"nevertheless", "next", "nine", "no", "nobody", "none", "noone",
			"nor", "not", "nothing", "now", "nowhere", "of", "off", "often", 
			"on", "once", "one", "only", "onto", "or", "other", "others", 
			"otherwise", "our", "ours", "ourselves", "out", "over", "own",
			"part", "per", "perhaps", "please", "put", "rather", "re", "same",
			"see", "seem", "seemed", "seeming", "seems", "serious", "several", 
			"she", "should", "show", "side", "since", "sincere", "six", "sixty",
			"so", "some", "somehow", "someone", "something", "sometime", "sometimes",
			"somewhere", "still", "such", "system", "take", "ten", "than", "that", 
			"the", "their", "them", "themselves", "then", "thence", "there", "thereafter",
			"thereby", "therefore", "therein", "thereupon", "these", "they", "thick",
			"thin", "third", "this", "those", "though", "three", "through", "throughout",
			"thru", "thus", "to", "together", "too", "top", "toward", "towards", 
			"twelve", "twenty", "two", "un", "under", "until", "up", "upon", "us", 
			"very", "via", "was", "we", "well", "were", "what", "whatever", "when",
			"whence", "whenever", "where", "whereafter", "whereas", "whereby", 
			"wherein", "whereupon", "wherever", "whether", "which", "while", 
			"whither", "who", "whoever", "whole", "whom", "whose", "why", "will",
			"with", "within", "without", "would", "yet", "you", "your", "yours", 
			"yourself", "yourselves", "the"};
}
