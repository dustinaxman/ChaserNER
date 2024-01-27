module.exports = {
  generatePayload: generatePayload
};

function generatePayload(context, events, done) {
  // Generate dynamic data for each request
  context.vars['text'] = 'Dynamic message ' + Math.random();

  // Continue with executing the scenario
  return done();
}