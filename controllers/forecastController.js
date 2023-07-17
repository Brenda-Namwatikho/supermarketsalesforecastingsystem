const { exec } = require('child_process');

function forecast(req, res) {
  const product_name = req.params.product_name;

  const pythonProcess = exec(`python ./preprocess_module.py ${product_name}`, (error, stdout, stderr) => {
    if (error) {
      console.error(`Error executing Python script: ${error.message}`);
      res.status(500).send('Internal Server Error');
      return;
    }

    if (stderr) {
      console.error(`Python script error: ${stderr}`);
      res.status(500).send('Internal Server Error');
      return;
    }

    const pythonOutput = stdout.trim();
    const jsonData = JSON.parse(pythonOutput);

    res.setHeader('Content-Type', 'application/json'); // Set the content type to JSON
    res.status(200).json(jsonData); // Send the JSON data as the response
  });

  pythonProcess.on('error', (error) => {
    console.error(`Error executing Python process: ${error.message}`);
    res.status(500).send('Internal Server Error');
  });
}

module.exports = {
  forecast
};
