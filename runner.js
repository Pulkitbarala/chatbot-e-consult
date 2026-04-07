// Demo runner for the summarizer. Run with: node sumarizer/runner.js

const { generateCommentSummary, createInteractiveSummarizerSession } = require('./summarizer');
let hfProvider;
try {
  hfProvider = require('./providers/huggingface').createHuggingFaceProvider();
} catch (e) {
  hfProvider = null;
}

// Sample data (replace with dashboard data when integrating)
const comments = [
  { text: 'The appointment scheduling was quick and easy.', sentiment: 'positive' },
  { text: 'Response times are too slow; I waited two days.', sentiment: 'negative' },
  { text: 'Interface looks clean but sometimes confusing on mobile.', sentiment: 'neutral' },
  { text: 'Great doctor notes, very clear and helpful.', sentiment: 'positive' },
  { text: 'Payment failed twice, very frustrating experience.', sentiment: 'negative' },
  { text: 'Notifications arrive late; I missed an update.', sentiment: 'negative' },
  { text: 'Video call quality was good.', sentiment: 'positive' },
];

const filter = process.argv[2] || 'all'; // 'positive only' | 'negative only' | 'all'

(async () => {
  const options = { provider: hfProvider };
  const result = await generateCommentSummary(comments, filter, options);
  console.log(result.text);

  // Interactive examples
  const session = await createInteractiveSummarizerSession(comments, filter, options);
  console.log('\n--- Follow-ups ---');
  console.log(session.generateMoreDetailedSummary());
  console.log('\n' + session.comparePositiveVsNegative());
  console.log('\nExplain point 1:', session.explainPoint(1));
})();
