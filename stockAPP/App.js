import React, { Component } from 'react';
import { WebView } from 'react-native-webview';

// ...
class MyWebComponent extends Component {
  render() {
    return (
      <WebView source={{ uri: 'http://localhost:5000' }} />
    );
  }
}

export default MyWebComponent
